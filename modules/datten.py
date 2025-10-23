import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from math import ceil
from torch import einsum

class Attention(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D,bias=bias)]

        if act == 'gelu': 
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K,bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self,x,no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)
        
        if no_norm:
            return x,A_ori
        else:
            return x,A

class AttentionGated(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

    def forward(self, x,no_norm=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        if no_norm:
            return x,A_ori
        else:
            return x,A

class DAttention(nn.Module):
    def __init__(self,input_dim=512,act='relu',gated=False,bias=False,dropout=False):
        super(DAttention, self).__init__()
        self.gated = gated
        if gated:
            self.attention = AttentionGated(input_dim,act,bias,dropout)
        else:
            self.attention = Attention(input_dim,act,bias,dropout)


 # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None,len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        assert ids_shuffle is not None

        _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,no_norm=False,mask_enable=False):

        if mask_enable and mask_ids is not None:
            x, _,_ = self.masking(x,mask_ids,len_keep)

        x,attn = self.attention(x,no_norm)

        if return_attn:
            return x.squeeze(1),attn.squeeze(1)
        else:   
            return x.squeeze(1)
        
        
      
def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-2], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

def exists(val):
    return val is not None
 

class CTCA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=False,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.heads = heads
        self.scale = dim_head ** -0.5

        # Separate linear layers for k, v and q
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # For k and v
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # Separate linear for q

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, q_input, mask=None, return_attn=False):


        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

        # Generate keys and values from x
        kv = self.to_kv(x)
        k, v = kv.chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (k, v))

        # Generate queries from q_input
        q = self.to_q(q_input)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        q = q * self.scale

        # Calculate the number of padded elements needed for landmarks


        # Generate landmarks
        l = ceil(n / m)
        k_landmarks = k.reshape(b, h, -1, l, k.size(-1)).sum(dim=-2)
        #v_landmarks = v.reshape(b, h, -1, l, v.size(-1)).sum(dim=-2)

        # Handle masking
        if mask is not None:
            mask = F.pad(mask, (0, padding), value=False) if remainder > 0 else mask
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))
            mask_landmarks = mask.reshape(b, 1, -1, l).sum(dim=-2) > 0

        # Calculate similarities
        sim1 = einsum('b h i d, b h j d -> b h i j', q, k_landmarks)
        sim2 = einsum('b h i d, b h j d -> b h i j', q, k_landmarks)
        sim3 = einsum('b h i d, b h j d -> b h i j', q, k)

        if mask is not None:
            mask_value = -torch.finfo(q.dtype).max
            sim1 = torch.where(mask[..., None] * mask_landmarks[..., None, :], sim1, mask_value)
            sim2 = torch.where(mask_landmarks[..., None] * mask_landmarks[..., None, :], sim2, mask_value)
            sim3 = torch.where(mask_landmarks[..., None] * mask[..., None, :], sim3, mask_value)

        # Apply softmax
        attn1, attn2, attn3 = map(lambda t: F.softmax(t, dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        # Aggregate values
        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # Apply convolutional residual connection
        if self.residual:
            out += self.res_conv(v.reshape(b, h, n + padding_length, -1)).reshape(b, h, n + padding_length, -1)

        # Combine heads and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if return_attn:
            return out, attn1 @ attn2_inv @ attn3

        return out

import math
import os
from architecture import clam 
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from architecture.network import Classifier_1fc, DimReduction, DimReduction1
from einops import repeat
from .nystrom_attention import NystromAttention
from modules.emb_position import *
from modules import transmil,WIKG,rrt

def pos_enc_1d(D, len_seq):
    
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                         -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLP_single_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_single_layer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class TransformWrapper1(nn.Module):
    def __init__(self, conf):
        super(TransformWrapper1, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = MutiHeadAttention2(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, conf.n_token, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class

        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, 0.0))
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input, use_attention_mask=True):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        outputs = []
        attns = []
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v, use_attention_mask=use_attention_mask)
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i)
        attns = torch.cat(attns, 1)
        feat_bag = self.bag_attention(v, attns.softmax(dim=-1).mean(1, keepdim=True))
        return torch.cat(outputs, dim=0), self.Slide_classifier(feat_bag), attns
    
class TransformWrapper(nn.Module):
    def __init__(self, conf):
        super(TransformWrapper, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.sub_attention = nn.ModuleList()
        for i in range(conf.n_token):
            self.sub_attention.append(MutiHeadAttention(conf.D_inner, 8, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop))
        self.bag_attention = MutiHeadAttention1(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, conf.n_token, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, 0.0))
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input, use_attention_mask=True):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        outputs = []
        attns = []
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v, use_attention_mask=use_attention_mask)
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i)
        attns = torch.cat(attns, 1)
        feat_bag = self.bag_attention(v, attns.softmax(dim=-1).mean(1, keepdim=True))
        return torch.cat(outputs, dim=0), self.Slide_classifier(feat_bag), attns


class MutiHeadAttention2(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        n_masked_patch: int = 0,
        mask_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj1 = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.out_proj1 = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, use_attention_mask=False) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        v1 = self.v_proj1(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            b, h, q, c = attn.shape
            n_masked_patch = min(self.n_masked_patch, c)
            _, indices = torch.topk(attn, n_masked_patch, dim=-1)
            indices = indices.reshape(b * h * q, -1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(b*h*q, c).to(attn.device)
            random_mask.scatter_(-1, masked_indices, 0)
            attn = attn.masked_fill(random_mask.reshape(b, h, q, -1) == 0, -1e9)

        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.layer_norm(out1)

        return out1[0], attn_out[0]

class MutiHeadAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        n_masked_patch: int = 0,
        mask_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, use_attention_mask=False) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            b, h, q, c = attn.shape
            n_masked_patch = min(self.n_masked_patch, c)
            _, indices = torch.topk(attn, n_masked_patch, dim=-1)
            indices = indices.reshape(b * h * q, -1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(b*h*q, c).to(attn.device)
            random_mask.scatter_(-1, masked_indices, 0)
            attn = attn.masked_fill(random_mask.reshape(b, h, q, -1) == 0, -1e9)

        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0], attn_out[0]

class MutiHeadAttention1(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, v: Tensor, attn: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(v)

        # Separate into heads
        v = self._separate_heads(v, self.num_heads)

        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0]



class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh())
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        return A  ### K x N
    
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    
class DAttention(nn.Module):
    def __init__(self,conf,n_classes,dropout,act):
        super(DAttention, self).__init__()
        self.L = 512 #512
        self.D = 128 #128
        self.K = 1
        self.feature = [nn.Linear(conf.D_feat, 512)]
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]
        if dropout:
            self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K))
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),)
        self.apply(initialize_weights)
        self.n_masked_patch=conf.n_masked_patch
        self.mask_drop=conf.mask_drop
    def forward(self, x,use_attention_mask=False):
        feature = self.feature(x)
        if len(feature.shape)==3:
            feature = feature.squeeze(0)
        elif len(feature.shape)==2:
            pass
        else:
            KeyError
        
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        '''
        if self.n_masked_patch>0 and use_attention_mask :
            # Get the indices of the top-k largest values
            k, n = A.shape
            #n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n, dim=-1)
            k1 = int(np.ceil((n*self.n_masked_patch)))
            _,cls_attn_topk_idx = torch.topk(A,k1,largest=True)
            cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)
            random_idx = torch.randperm(cls_attn_topk_idx.size(0),device=cls_attn_topk_idx.device)
            if self.mask_drop !=0:
                rand_selected = torch.gather(cls_attn_topk_idx,dim=0,index=random_idx[:int(np.ceil((cls_attn_topk_idx.size(0)*self.mask_drop)))])
                #rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n*self.n_masked_patch * self.mask_drop)]
                masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
                random_mask = torch.ones(k, n).to(A.device)
                random_mask.scatter_(-1, masked_indices, 0)
                A = A.masked_fill(random_mask == 0, -1e9)
            else:
                A=A
        '''
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        Y_prob = self.classifier(M)
        return Y_prob,M
     

class AttnMIL(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, 1)
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
    def forward(self, x,return_attn=False): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = self.classifier(afeat)
        if return_attn:
            return outputs,A_out
        else:
            return outputs





class AttnMIL1(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL1, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        #self.attention_scoce = Attention(conf.D_inner)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = conf.mask_drop

    def forward(self, x, use_attention_mask=False,pseudo_bag=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        #A2=self.attention_scoce(x)
        A = self.attention(x)  ## K x N
        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        # mask_drop = torch.zeros([0])
        # if self.n_masked_patch > 0 and use_attention_mask:
        #     # Get the indices of the top-k largest values
        #     k, n = A.shape
        #     n_masked_patch = min(self.n_masked_patch, n)
        #     _, indices = torch.topk(A, n_masked_patch, dim=-1)
        #     confidence_values = torch.topk(torch.softmax(A, dim=1), n_masked_patch, dim=-1)[0]
        #     mask_drop = torch.clamp((confidence_values[:,:1].sum(dim=-1) / confidence_values[:,:2].sum(dim=-1) - 0.5) * 2, min=0.2, max=0.9)
        #     shuffled_index = torch.argsort(torch.rand(*indices.shape), dim=-1)
        #     for i in range(self.n_token):
        #         masked_indices = indices[i][shuffled_index[i][:int(mask_drop[i] * n_masked_patch)]]
        #         A[i,masked_indices] = -1e9

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, x) ## K x L
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        return torch.stack(outputs, dim=0), self.Slide_classifier(afeat.mean(dim=0, keepdim=True)), A_out.unsqueeze(0)


class AttnMIL4(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL4, self).__init__()
        self.dimreduction = DimReduction(conf.feat_d, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.mask_drop = conf.mask_drop

    def forward(self, x, is_train=True): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        if self.n_masked_patch > 0 and is_train:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = []
        # max_conf = []
        for i, head in enumerate(self.classifier):
            output = head(afeat[i])
            outputs.append(output)
            # max_conf.append(torch.softmax(output, dim=0).amax())
        outputs = torch.stack(outputs)
        return outputs, outputs.mean(dim=0, keepdim=True), A_out.unsqueeze(0)

class AttnMIL3(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL3, self).__init__()
        self.dimreduction = DimReduction(conf.feat_d, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        # self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = conf.mask_drop

    def forward(self, x, is_train=True): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        if self.n_masked_patch > 0 and is_train:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = []
        # max_conf = []
        for i, head in enumerate(self.classifier):
            output = head(afeat[i])
            outputs.append(output)
            # max_conf.append(torch.softmax(output, dim=0).amax())
        outputs = torch.stack(outputs)
        return outputs, outputs.max(axis=0)[0].unsqueeze(0), A_out.unsqueeze(0)



# AttnMIL5 基本没啥用
class AttnMIL5(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL5, self).__init__()
        self.dimreduction = DimReduction(conf.feat_d, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = conf.mask_drop

    def forward(self, x, is_train=True): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        if self.n_masked_patch > 0 and is_train:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = []
        # max_conf = []
        for i, head in enumerate(self.classifier):
            output = head(afeat[i])
            outputs.append(output)
            # max_conf.append(torch.softmax(output, dim=0).amax())
        outputs = torch.stack(outputs)
        return outputs, self.Slide_classifier(afeat).amax(dim=0).unsqueeze(0), A_out.unsqueeze(0)

class AttnMIL2(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL2, self).__init__()
        self.dimreduction = DimReduction(conf.feat_d, conf.D_inner)
        self.attention1 = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.attention2 = Attention_Gated(conf.D_inner, D, 1)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

    def forward(self, x, is_train=True): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention1(med_feat)  ## K x N

        if self.n_masked_patch > 0 and is_train:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * 0.2)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        A2 = self.attention2(afeat)
        A2 = F.softmax(A2, dim=1)
        afeat = torch.mm(A2, afeat)
        return torch.stack(outputs, dim=0), self.Slide_classifier(afeat), A_out.unsqueeze(0)


class AttnMIL6(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL6, self).__init__()
        self.dimreduction = DimReduction(conf.input_dim, conf.D_inner) 
        self.attention = Attention_Gated(conf.D_inner, D, 1)
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_classes, droprate)
        self.topk=conf.topk
    def forward(self, x, label=None,use_attention_mask=False,pseudo_bag=False,instance_eval=False,return_bag_feature=False, mask_drop=0.0): ## x: N x L
        if len(x.shape)==3:
            x=x.squeeze(0)
        elif len(x.shape)==2:
            pass
        else:
            KeyError
        
        x0=x
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N
        A_out_origin = A
        
        if  use_attention_mask and mask_drop!=0 :
            _, n = A.shape
          
            attn_score = A.mean(dim=0)  # shape: (n,)

            sorted_indices = torch.argsort(attn_score, descending=True)
 
            num_keep_high = int(n * self.topk)
            num_keep_high = max(1, num_keep_high)
            keep_high_indices = sorted_indices[:num_keep_high]

            remaining_indices = sorted_indices[num_keep_high:]  # 
            num_remaining = remaining_indices.shape[0]

            keep_ratio_low = 1-mask_drop  
            num_keep_low = int(num_remaining * keep_ratio_low)

            rand_perm = torch.randperm(num_remaining, device=A.device)
            keep_low_indices = remaining_indices[rand_perm[:num_keep_low]]

            final_keep_indices = torch.cat([keep_high_indices, keep_low_indices], dim=0)
            final_keep_indices = final_keep_indices.sort().values  

            A = A[:, final_keep_indices]
            x = x[final_keep_indices]        
        else:
            A=A
             # softmax over N
        A = F.softmax(A, dim=1) 
        afeat = torch.mm(A, x) ## K x L 5*256
        
        
        if pseudo_bag:
            return  self.Slide_classifier(afeat), A_out_origin.unsqueeze(0),afeat
        elif return_bag_feature:
            return self.Slide_classifier(afeat), A_out_origin.unsqueeze(0),afeat
        else:
            return  self.Slide_classifier(afeat), A_out_origin.unsqueeze(0)
    



class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head=8, n_token=1):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=head,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
            n_token=n_token,
        )

    def forward(self, x, need_attn=False):
        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=need_attn)
            x = x + z
            return x, attn
        else:
            x = x + self.attn(self.norm(x))
            return x



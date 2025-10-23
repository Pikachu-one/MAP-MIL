


import torch
from torch import nn
import torch.nn.functional as F

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
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        return A  ### K x N

class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class ACMIL(nn.Module):
    #def __init__(self, input_dim, n_class, D=128, droprate=0, n_masked_patch=10, n_token=5, mask_drop=0.6):
    def __init__(self, input_dim, n_class, D=128, droprate=0, n_masked_patch=10, n_token=5, mask_drop=0.0):
        super(ACMIL, self).__init__()
        self.dimreduction = DimReduction(input_dim, 256)
        self.attention = Attention_Gated(256, D, n_token)
        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(256, n_class, droprate))
        self.n_masked_patch = n_masked_patch
        self.n_token = n_token
        self.Slide_classifier = Classifier_1fc(256, n_class, droprate)
        self.mask_drop = mask_drop

    def forward(self, x, pseudo_bag=False):  ## x: N x L
        if x.dim() == 3:
            x = x[0]  # 1 * 1000 * 512
        elif x.dim() == 2:
            pass
        else:
            raise KeyError("Unexpected tensor dimension: {}".format(x.dim()))

        x = self.dimreduction(x)  # 1000 * 256
        A = self.attention(x)  ##  1000 * 1

        if self.n_masked_patch > 0 and self.training and self.mask_drop > 0:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            # n_masked_patch = int(n * 0.01)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, x)  ## K x L
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)

        if pseudo_bag:
            return torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_out.unsqueeze(0),bag_feat
        else:
            return torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_out.unsqueeze(0)


            
if __name__ == "__main__":
    import torch
    import yaml
    from thop import profile, clever_format  # 用于计算 FLOPs 和参数量
    
    x = torch.rand(1, 12255, 1024).cuda() 
    model = ACMIL(input_dim=1024, n_class=2).cuda()
    model.eval()

    # ---- 2. 计算参数量与 FLOPs ----
    flops, params = profile(model, inputs=(x,))  # 

    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Params: {params / 1e6:.3f} M parameters")

   
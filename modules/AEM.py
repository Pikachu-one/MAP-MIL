
import torch
from torch import nn
import torch.nn.functional as F

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


class ABMIL(nn.Module):
    def __init__(self, input_dim, n_class,D=128, droprate=0):
        super(ABMIL, self).__init__()
        self.dimreduction = DimReduction(input_dim, 256)
        self.attention = Attention_Gated(256, D, 1)
        self.classifier = Classifier_1fc(256, n_class, droprate)

    def forward(self, x, is_train=False):  ## x: N x L
        if x.dim()==3:
            x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat)  ## K x L
        outputs = self.classifier(afeat)
        return outputs, A_out.unsqueeze(0)

    def get_features(self, x):
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat)  ## K x L
        return afeat, A_out

if __name__ == "__main__":
    import torch
    import yaml
    from thop import profile, clever_format  # 用于计算 FLOPs 和参数量
    
    x = torch.rand(1, 12255, 1024).cuda()  
    model = ABMIL(input_dim=1024, n_class=2).to(device).cuda()
    model.eval()

    # ---- 2. 计算参数量与 FLOPs ----
    flops, params = profile(model, inputs=(x,))  # 

    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Params: {params / 1e6:.3f} M parameters")

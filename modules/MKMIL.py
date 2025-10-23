"""
MambaMIL_SRMamba
"""
import torch
import torch.nn as nn
from thop import profile

# from torch.nn import LayerNorm, ReLU
# from torch_geometric.nn import GENConv, DeepGCNLayer
# torch.use_deterministic_algorithms(False)

from mamba_ssm import SRMamba
from mamba_ssm import BiMamba
from mamba_ssm import Mamba
from mamba_ssm import AFWMamba
# from .fast_kan import Fast_KANLinear
import torch.nn.functional as F
# from thop import profile


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MKMIL(nn.Module):
    def __init__(self, n_classes, dropout, act, n_features=1024, layer=2, rate=10, type="SRMamba"):
        super(MKMIL, self).__init__()
        self._fc1 = [nn.Linear(n_features, 512), nn.Linear(512, 256)]
        # self._fc1 = [nn.Linear(n_features, 256)]
        self._fc2 = [nn.Linear(256, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
            self._fc2 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
            self._fc2 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
            self._fc2 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self._fc2 = nn.Sequential(*self._fc2)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(256),
                        SRMamba(
                            d_model=256,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(256),
                        Mamba(
                            d_model=256,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif type == "AFWMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(256),
                        AFWMamba(
                            d_model=256,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(256),
                        BiMamba(
                            d_model=256,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        # self.classifier = Fast_KANLinear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            # Fast_KANLinear(512,128),
            nn.Tanh(),
            # Fast_KANLinear(128, 1)
            nn.Linear(128, 1)
        )
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    def forward(self, x):

        h = x.float()  # [B, n, 1024]
        B, N, C = h.size()

        h = self._fc1(h)  # [B, n, 512]

        if self.type == "SRMamba" or self.type == "AFWMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)  # layerNorm
                h = layer[1](h, rate=self.rate)  # SRMamba   [B, n, 512]
                h = h + h_  # [B, n, 512]
        elif self.type == "Mamba" or self.type == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_
        h = self._fc2(h)
        h = self.norm(h)
        A = self.attention(h)  # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A_1 = A
        A = F.softmax(A, dim=-1)  # [B, K, n]
        h = torch.bmm(A, h)  # [B, K, 512]
        h = h.squeeze(0)
        # ---->predict
        logits = self.classifier(h)  # [B, n_classes]
        
        return logits


            
if __name__ == "__main__":
    import torch
    import yaml
    from thop import profile, clever_format  # 用于计算 FLOPs 和参数量
    
    x = torch.rand(1, 12255, 1024).cuda()  # 模拟输入
    model = MKMIL(n_classes=2, dropout=0.5, act='relu', n_features=1024, layer=2, rate=10, type="AFWMamba").cuda()
    model.eval()

    # ---- 2. 计算参数量与 FLOPs ----
    flops, params = profile(model, inputs=(x,))  # 

    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Params: {params / 1e6:.3f} M parameters")

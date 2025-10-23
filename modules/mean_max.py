import torch.nn as nn

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

class MeanMIL(nn.Module):
    def __init__(self,conf,dropout=True,act='relu',test=False):
        super(MeanMIL, self).__init__()

        head = [nn.Linear(1024,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
            
        head += [nn.Linear(512,conf.n_class)]
        
        self.head = nn.Sequential(*head)

        self.apply(initialize_weights)

    def forward(self,x):
        x = self.head(x).mean(axis=1)
        return x

class MaxMIL(nn.Module):
    def __init__(self,conf,dropout=True,act='relu',test=False):
        super(MaxMIL, self).__init__()

        head = [nn.Linear(1024,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]

        head += [nn.Linear(512,conf.n_class)]
        self.head = nn.Sequential(*head)

        self.apply(initialize_weights)

    def forward(self,x):
        x,_ = self.head(x).max(axis=1)
        return x
if __name__ == "__main__":
    import torch
    import yaml
    from thop import profile, clever_format  # 用于计算 FLOPs 和参数量
    
    x = torch.rand(1, 12255, 1024).cuda()  # 模拟输入
    model = MaxMIL().cuda()
    model.eval()

    # ---- 2. 计算参数量与 FLOPs ----
    flops, params = profile(model, inputs=(x,))  # 

    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Params: {params / 1e6:.3f} M parameters")

    x = torch.rand(1, 12255, 1024).cuda()  # 模拟输入
    model = MeanMIL().cuda()
    model.eval()

    # ---- 2. 计算参数量与 FLOPs ----
    flops, params = profile(model, inputs=(x,))  # 

    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Params: {params / 1e6:.3f} M parameters")
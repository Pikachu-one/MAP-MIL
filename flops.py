import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from architecture.transformer import AttnMIL6
from architecture.network import Classifier_1fc, DimReduction, DimReduction1
import numpy as np
''''
class AttnMIL6(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL6, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner) 
        self.attention = Attention_Gated(conf.D_inner, D, 1)
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
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
 
            num_keep_high = int(n * 0.05)
            num_keep_high = max(1, num_keep_high)
    
            keep_high_indices = sorted_indices[:num_keep_high]

            remaining_indices = sorted_indices[num_keep_high:]  
            num_remaining = remaining_indices.shape[0]

            keep_ratio_low = 1-mask_drop  # 例如保留低分中的一半
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
'''
class MAP_MILModel(nn.Module):
   
    def __init__(self, base_milnet, conf):
        super().__init__()
        self.milnet = base_milnet
        self.conf = conf
        self.criterion = nn.CrossEntropyLoss()
        self.numGroup =6
        self.high_medium_line=0.05
        self.medium_low_line=0.3
        self.mask_drop=0.25
        self.diff_loss='KL_loss'
    def forward(self, image_patches, labels=None, mode='train'):
        """
        Args:
            image_patches: (1, N, 1024)
            labels: (1,)
            mode: 'train' or 'test'
        """
        device = image_patches.device

        slide_preds, attn, bag_feature = self.milnet(
            image_patches, 
            use_attention_mask=True if mode=='train' else False,
            return_bag_feature=True, 
            mask_drop=self.mask_drop if mode=='train' else 0.0,
        )
        loss_cls = self.criterion(slide_preds, labels) if labels is not None else 0.0

        _, n = attn.squeeze(0).shape
        n0 = n

        if mode == 'test' or n0 < self.numGroup * 3:
            return {'loss': loss_cls, 'pred': slide_preds, 'attn': attn}

        image_patches = image_patches.squeeze(0)
        _, indices = torch.topk(attn, n, dim=-1)
        indices = indices.flatten().cpu().numpy()

        split_1 = int(n0 * self.high_medium_line)
        split_2 = int(n0 * self.medium_low_line)
        split_3 = n0

        top = indices[:split_1]
        mid = indices[split_1:split_2]
        low = indices[split_2:split_3]

        top_groups = np.array_split(top, self.numGroup)
        mid_groups = np.array_split(mid, self.numGroup)
        low_groups = np.array_split(low, self.numGroup)

        pseudo_bags = []
        for t, m, l in zip(top_groups, mid_groups, low_groups):
            pseudo_idx = np.concatenate([t, m, l])
            pseudo_bags.append(image_patches[pseudo_idx])

        pseudo_bag_fea_list = []
        pseudo_bag_logit_list = []
        label_groups = [labels for _ in range(self.numGroup)]
        total_pseudo_loss = 0.0

        for pseudo_bag, label in zip(pseudo_bags, label_groups):
            pseudo_bag = pseudo_bag.unsqueeze(0)
            slide_preds_p, attn_p, pseudo_bag_fea = self.milnet(
                pseudo_bag, use_attention_mask=False, pseudo_bag=True
            )
            pseudo_bag_fea_list.append(pseudo_bag_fea)
            pseudo_bag_logit_list.append(slide_preds_p)
            total_pseudo_loss += self.criterion(slide_preds_p.view(1, -1), label)

        bag_loss = total_pseudo_loss / self.numGroup
        pseudo_bag_fea_sum = torch.cat(pseudo_bag_fea_list, dim=0)
        pseudo_bag_logit_sum = torch.cat(pseudo_bag_logit_list, dim=0)

        # === 5️⃣ diff_loss ===
        pseudo_num, _ = pseudo_bag_fea_sum.shape
        diff_loss = torch.tensor(0.0, device=device)
        for i in range(pseudo_num):
            for j in range(i + 1, pseudo_num):
                if self.diff_loss == 'MSE':
                    diff_loss += F.mse_loss(
                        pseudo_bag_fea_sum[i], pseudo_bag_fea_sum[j]
                    ) / (pseudo_num * (pseudo_num - 1) / 2)
                elif self.diff_loss == 'cosine':
                    diff_loss += 1 - torch.cosine_similarity(
                        pseudo_bag_fea_sum[i], pseudo_bag_fea_sum[j], dim=-1
                    ).mean() / (pseudo_num * (pseudo_num - 1) / 2)
                elif self.diff_loss == 'KL_loss':
                    diff_loss += F.kl_div(
                        F.log_softmax(pseudo_bag_logit_sum[i], dim=-1),
                        F.softmax(pseudo_bag_logit_sum[j], dim=-1),
                        reduction='batchmean'
                    )

        loss = loss_cls + bag_loss + diff_loss

        return {
            'loss': loss,
            'cls_loss': loss_cls,
            'bag_loss': bag_loss,
            'diff_loss': diff_loss,
            'pred': slide_preds,
            'attn': attn
        }

        
if __name__ == "__main__":
    import torch
    import yaml
    from thop import profile, clever_format  # 用于计算 FLOPs 和参数量
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
   
    # ---- 1. 读取配置并实例化模型 ----
    with open('./config/GastricCancer_natural_supervised_config.yml', "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        conf = Struct(**c)
        
    x = torch.rand(1, 12255, 1024).cuda()  # 模拟输入
    
    base_milnet = AttnMIL6(conf).cuda()
    model = MAP_MILModel(base_milnet, conf).cuda()

    flops, params = profile(model, inputs=(x, torch.tensor([1]).cuda(), 'train'), verbose=False)
    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Params: {params / 1e6:.3f} M parameters")
    
    model.eval()
    # ---- 2. 计算参数量与 FLOPs ----
    flops, params = profile(model, inputs=(x,None,'test'))  # 

    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Params: {params / 1e6:.3f} M parameters")
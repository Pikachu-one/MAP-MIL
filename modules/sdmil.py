from torch import nn
from modules.emb_position import *
from modules.datten import *
from modules.rmsa import *
from .nystrom_attention import NystromAttention
from modules.datten import DAttention
from timm.models.layers import DropPath
from einops import rearrange, reduce
from numpy.linalg import norm
from modules import mean_max,clam,transmil,dsmil

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8,drop_out=0.1,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,attn='rmsa',n_region=8,epeg=False,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,crmsa_k=3,epeg_k=15,**kwargs):
        super().__init__()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        if attn == 'ntrans':
            self.attn = NystromAttention(
                dim = dim,
                dim_head = trans_dim,  # dim // 8
                heads = head,
                num_landmarks = 256,    # number of landmarks dim // 2
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=drop_out)
        elif attn == 'rmsa':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs)
        elif attn == 'crmsa':
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs
            )
        else:
            raise NotImplementedError

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_out) if ffn else nn.Identity()

    def forward(self,x,coord=None,need_attn=False):

        x,attn = self.forward_trans(x,coord=coord,need_attn=need_attn)
        
        if need_attn:
            return x,attn
        else:
            return x

    def forward_trans(self, x,coord=None,need_attn=False):
        attn = None
        
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x+self.drop_path(z)
        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x,attn
class RRTEncoder(nn.Module):
    def __init__(self,mlp_dim=512,pos_pos=0,pos='none',peg_k=7,attn='rmsa',region_num=8,drop_out=0.1,n_layers=2,n_heads=8,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,epeg=True,epeg_k=15,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,peg_bias=True,peg_1d=False,cr_msa=True,crmsa_k=3,all_shortcut=False,crmsa_mlp=False,crmsa_heads=8,need_init=False,**kwargs):
        super(RRTEncoder, self).__init__()
        self.final_dim = mlp_dim
        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut

        self.layers = TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn=attn,n_region=8,epeg=epeg,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,epeg_k=epeg_k,**kwargs)
        self.layers_12 = TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn=attn,n_region=12,epeg=epeg,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,epeg_k=epeg_k,**kwargs)
        self.layers_16 = TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn=attn,n_region=16,epeg=epeg,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,epeg_k=epeg_k,**kwargs)
        # only for ablation
        
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        elif pos =='APE':
            self.pos_embedding=APE(embed_dim=mlp_dim)
        elif pos =='coord':
            self.pos_embedding=Coord(embed_dim=mlp_dim)
        else:
            self.pos_embedding = nn.Identity()
        
        self.pos=pos
        self.pos_pos = pos_pos
        if need_init:
            self.apply(initialize_weights)
       
    def forward(self, x,coord=None): 
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0),x.size(1),-1)
            x = x.transpose(1,2)
            shape_len = 4
        batch, num_patches, C = x.shape   
        
        if self.pos_pos == -1:
            if self.pos=='coord':
                x = self.pos_embedding(x,coord)
            else:
                x = self.pos_embedding(coord)
        # R-MSA within region
        
        x_shortcut2 = x   
        x1 = self.layers(x,coord)
        x2 = self.layers_12(x,coord)
        x3 = self.layers_16(x,coord)
        if self.all_shortcut:
            x = x1+x_shortcut2+x2+x3                       
        x = self.norm(x)
        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1,2)
            x = x.reshape(batch,C,int(num_patches**0.5),int(num_patches**0.5))
        return x
    
def distance_calculate(x,prototype):
    B,N,C=x.shape
    if len(x.shape)==3:
        x=x.squeeze(0) 
        
    if x.shape[0] == 1:
        weights = torch.ones((1, x.shape[0]), device=x.device)  # shape: (1, 1)
        return x, weights
    else:
        x_expanded = x.unsqueeze(0).expand(prototype.shape[0], -1, -1)  # (P, N, C)
        prototype_expanded = prototype.unsqueeze(1).expand(-1, x.shape[0], -1)  # (P, N, C)
        
        ## 计算欧氏距离
        distances = torch.sqrt(torch.sum((x_expanded - prototype_expanded) ** 2, dim=2))  # (P, N)
        # 计算曼哈顿距离
        #distances = torch.sum(torch.abs(x_expanded - prototype_expanded), dim=2)
        '''
        # 计算范数  
        x_norm = torch.norm(x_expanded, dim=2, keepdim=True).clone()  # (P, N, 1)
        prototype_norm = torch.norm(prototype_expanded, dim=2, keepdim=True).clone()   # (P, N, 1)
        # 避免除以零的情况
        x_norm[x_norm == 0] = 1
        prototype_norm[prototype_norm == 0] = 1
        # 计算余弦相似度
        cosine_similarity = torch.sum(x_expanded * prototype_expanded, dim=2) / (x_norm * prototype_norm).squeeze(2)  # (P, N)
        # 计算余弦距离
        distances = 1 - cosine_similarity
        '''
        # 对距离进行归一化
        distances=distances/torch.sqrt(torch.tensor(x.shape[1])) #    
        # 计算每个实例的权重
        inv_distances = 1.0 / distances  # (P, N)
        sum_inv_distances = torch.sum(inv_distances, dim=1, keepdim=True)  # (P, 1)
        weights = inv_distances / sum_inv_distances   # (P, N)
        x=weights.transpose(0,1)*x
        #x = torch.mm(weights, x)
        return x,weights


def similarity_calculate(x,prototype):
    B,N,C=x.shape
    if len(x.shape)==3:
        x=x.squeeze(0) 
    # 计算余弦相似度
    prototype_norm = torch.norm(prototype, dim=1, keepdim=True)
    x_norm = torch.norm(x, dim=1, keepdim=True)
    # 避免除以零的情况
    prototype_norm[prototype_norm == 0] = 1
    x_norm[x_norm == 0] = 1
    # 计算相似度
    similarities = torch.matmul(x, prototype.T) / (x_norm * prototype_norm)
    # 计算权重
    total_similarity = torch.sum(similarities)
    weights = similarities / total_similarity  #weights.shape=[N,1]
    x = weights * x 
    return x

def attention_calculate(x,Prototype,confounder_merge='cat'):
        B,N,C=x.shape
        if len(x.shape)==3:
            x=x.squeeze(0) 
        if confounder_merge=='cat':
            W_q = nn.Linear(512, 256).cuda()
            W_k = nn.Linear(512, 256).cuda()
            x=W_q(x)
            Prototype=W_k(Prototype)
            deconf_A = torch.mm(x, Prototype.transpose(0, 1)) #N*1
            deconf_A = F.softmax( deconf_A / torch.sqrt(torch.tensor(x.shape[1])), 0)
            #print('deconf_A.shape:',deconf_A.shape)
            # 将 deconf_A 扩展到与 x 相同的形状
            deconf_A_expanded = deconf_A.expand_as(x)
            #print('x.shape:',x.shape)
            #print('deconf_A_expanded.shape:',deconf_A_expanded.shape)
            # 对应权重乘到对应实例上
            x_weight = x * deconf_A_expanded
            #print('x_weight.shape:',x_weight.shape)
            x=torch.cat((x,x_weight),dim=1)
            #print('conca_x.shape:',x_weight.shape)
            return x
        elif confounder_merge=='add':
            x=W_q(x)
            Prototype=W_k(x)
            deconf_A = torch.mm(x, Prototype.transpose(0, 1)) #N*1
            deconf_A = F.softmax( deconf_A / torch.sqrt(torch.tensor(x.shape[1])), 0)
            # 将 deconf_A 扩展到与 x 相同的形状
            deconf_A_expanded = deconf_A.unsqueeze(1).expand_as(x)
            # 对应权重乘到对应实例上
            x_weight = x * deconf_A_expanded
            x=x+x_weight
        elif  confounder_merge=='sub':
            x=W_q(x)
            Prototype=W_k(x)
            deconf_A = torch.mm(x, Prototype.transpose(0, 1)) #N*1
            deconf_A = F.softmax( deconf_A / torch.sqrt(torch.tensor(x.shape[1])), 0)
            # 将 deconf_A 扩展到与 x 相同的形状
            deconf_A_expanded = deconf_A.unsqueeze(1).expand_as(x)
            # 对应权重乘到对应实例上
            x_weight = x * deconf_A_expanded
            x=x-x_weight
        else:
            raise ValueError
class SDMIL(nn.Module):
    def __init__(self, input_dim=1024,mlp_dim=512,act='relu',n_classes=2,dropout=0.25,pos_pos=0,pos='none',peg_k=7,attn='rmsa',pool='attn',region_num=16,n_layers=2,n_heads=8,drop_path=0.,da_act='relu',trans_dropout=0.1,ffn=False,ffn_act='gelu',mlp_ratio=4.,da_gated=True,da_bias=False,da_dropout=False,trans_dim=64,epeg=True,min_region_num=0,qkv_bias=True,model_dowan_aggregate_model='abmil',**kwargs):
        super(SDMIL, self).__init__()
        self.model_dowan_aggregate_model=model_dowan_aggregate_model
        print('self.model_dowan_aggregate_model:',self.model_dowan_aggregate_model)
        self.patch_to_emb = [nn.Linear(input_dim, 512)]
        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]
        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)
        self.online_encoder = RRTEncoder(mlp_dim=mlp_dim,pos_pos=pos_pos,pos=pos,peg_k=peg_k,attn=attn,region_num=region_num,n_layers=n_layers,n_heads=n_heads,drop_path=drop_path,drop_out=trans_dropout,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,epeg=epeg,min_region_num=min_region_num,qkv_bias=qkv_bias,**kwargs)
        #gatemil
        #abmil
        if self.model_dowan_aggregate_model=='abmil':
            self.pool_fn = DAttention(self.online_encoder.final_dim,da_act,gated=False,bias=da_bias,dropout=da_dropout) if pool == 'attn' else nn.AdaptiveAvgPool1d(1)
        elif self.model_dowan_aggregate_model=='gatemil':
            self.pool_fn = DAttention(self.online_encoder.final_dim,da_act,gated=True,bias=da_bias,dropout=da_dropout) if pool == 'attn' else nn.AdaptiveAvgPool1d(1)
        elif self.model_dowan_aggregate_model=='clam_mb':
            self.pool_fn=clam.CLAM_MB(input_dim=self.online_encoder.final_dim,n_classes=n_classes,dropout=da_dropout,act=act)
        elif  self.model_dowan_aggregate_model=='clam_sb':
            self.pool_fn=clam.CLAM_SB(input_dim=self.online_encoder.final_dim,n_classes=n_classes,dropout=da_dropout,act=act)
        elif self.model_dowan_aggregate_model=='transmil':
            self.pool_fn=transmil.TransMIL(input_dim=self.online_encoder.final_dim,n_classes=n_classes,dropout=da_dropout,act=act)
        elif self.model_dowan_aggregate_model=='dsmil':
            self.pool_fn=dsmil.MILNet(input_dim=self.online_encoder.final_dim,n_classes=1,dropout=da_dropout,act=act)

        #self.att = DAttention(self.online_encoder.final_dim,da_act,gated=da_gated,bias=da_bias,dropout=da_dropout) if pool == 'attn' else nn.AdaptiveAvgPool1d(1)
        self.att = AttentionGated(self.online_encoder.final_dim,act=da_act,bias=da_bias,dropout=da_dropout) 
        self.predictor = nn.Linear(self.online_encoder.final_dim,n_classes)
        self.predictor2 = nn.Linear(self.online_encoder.final_dim,n_classes)
        #self.mean_prototype=mean_max.MeanMIL(input_dim=self.online_encoder.final_dim,n_classes=n_classes,dropout=dropout,act=act,rrt=None)
        #self.max_prototype=mean_max.MaxMIL(input_dim=self.online_encoder.final_dim,n_classes=n_classes,dropout=dropout,act=act,rrt=None)
        
        #self.pos_embedding = PEG(mlp_dim,k=peg_k,bias=True,conv_1d=False)
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #初始化
        self.fuse_weight_1.data.fill_(1/3)
        self.fuse_weight_2.data.fill_(1/3)
        self.fuse_weight_3.data.fill_(1/3)
        self.apply(initialize_weights)
    
    def forward(self, x,label=None,loss=None,train=False,return_attn=False,no_norm=False,prototype_feature=False,pseudo_bag=False):
        #weights = F.softmax(torch.cat([self.fuse_weight_1, self.fuse_weight_2], dim=0), dim=0)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size,num_instances,C=x.shape
        x = self.patch_to_emb(x) # n*512
        x = self.dp(x)
        # feature re-embedding
        x = self.online_encoder(x)
        
        
        #x0=x
        #att为abmil
        #Prototype=self.att(x)

        #att为gatemmil
        Prototype,a=self.att(x)
        Prototype=Prototype.squeeze(0)

            
        #share abmil
        #Prototype,a=self.pool_fn(x,return_attn=return_attn)

        #mean_prototype
        #Prototype=self.mean_prototype(x)

        #max_prototype
        #Prototype=self.max_prototype(x)
        #核心步骤
        #x_self_att_fea=attention_calculate(x,Prototype,confounder_merge)
        #x_distance_feature=similarity_calculate(x,Prototype)
        x_distance_feature,weights =distance_calculate(x,Prototype)
     
        #x0=x_distance_feature
        #add 
        #x=(self.fuse_weight_1*x+self.fuse_weight_2*x_distance_feature) 
        x=(self.fuse_weight_1*x+self.fuse_weight_2*x_distance_feature) 
        

        #x=(x+x_distance_feature) 
        
        #cat
        #x1=self.fuse_weight_1*x
        #x2=self.fuse_weight_2*x_distance_feature
        #x1=x
        #x2=x_distance_feature
        #x2=x2.unsqueeze(0)
        #x = torch.cat((x1, x2), dim=2)  # 在特征维度上拼接
        #x=self.patch_to_emb(x)
        #sub 
        #x=(self.fuse_weight_1*x-self.fuse_weight_2*x_distance_feature) 
        #x=x-x_distance_feature
        # feature aggregation
        if self.model_dowan_aggregate_model in ('abmil','gatemil','transmil'):
        #abmil/gatemil
            #保证包级特征辅助loss
            #auxiliary_logits=self.predictor2(Prototype)
            if return_attn:
                x,A= self.pool_fn(x,return_attn=return_attn)
            else:
                x= self.pool_fn(x,return_attn=return_attn)
            #Prototype1=x
            logits = self.predictor(x)
            if train:
                return logits
            elif prototype_feature:
                return x,Prototype
            elif return_attn:
                return logits,a,A
            elif pseudo_bag:
                return logits,x
            else:
                return logits
            #CLAM_MB CLAM_SB
        elif self.model_dowan_aggregate_model=='clam_mb'or self.model_dowan_aggregate_model=='clam_sb':
            #保证包级特征辅助loss
            #logits,total_inst_loss,ps=self.pool_fn(x,label=label,return_att=return_attn)
            logits,total_inst_loss,ps=self.pool_fn(x,label=label)
            if train:
                return logits,total_inst_loss,ps
            else:
                return logits
        elif self.model_dowan_aggregate_model=='dsmil':
            #保证包级特征辅助loss
            logits,cls_loss,patch_num=self.pool_fn(x,label=label,loss=loss)
            if train:
                return logits,cls_loss,patch_num
            else:
                return logits,cls_loss
        elif self.model_dowan_aggregate_model=='dftd':
            #保证包级特征辅助loss
            return x
        else:
            ValueError
       
            

      
        
if __name__ == "__main__":
    import torch
    import yaml
    from thop import profile, clever_format  # 用于计算 FLOPs 和参数量
    
    x = torch.rand(1, 12255, 1024).cuda()  # 模拟输入
    
    model_params = {
            'input_dim': 1024,
            'n_classes': 2,
            'dropout': 0.25,
            'act': 'relu',
            'region_num': 8,
            'pos': 'none',
            'pos_pos': 0,
            'pool': 'attn',
            'peg_k': 7,
            'drop_path': 0.0,
            'n_layers': 2,
            'n_heads': 8,
            'attn': 'rmsa',
            'da_act': 'tanh',
            'trans_dropout': 0.1,
            'ffn': False,
            'mlp_ratio': 4.0,
            'trans_dim': 64,
            'epeg': True,
            'min_region_num': 0,
            'qkv_bias': True,
            'epeg_k': 21,
            'epeg_2d': False,
            'epeg_bias': True,
            'epeg_type': 'attn',
            'region_attn': 'native',
            'peg_1d': False,
            'cr_msa': True,
            'crmsa_k': 5,
            'all_shortcut': True,
            'crmsa_mlp':False,
            'crmsa_heads':1,}
    model = sdmil.SDMIL(**model_params).cuda()
    
    model.eval()

    # ---- 2. 计算参数量与 FLOPs ----
    flops, params = profile(model, inputs=(x,))  # 

    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Params: {params / 1e6:.3f} M parameters")

# !/usr/bin/env python
import sys
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from engine import loss_forward_and_backward
from utils.utils import save_model, Struct, set_seed, Wandb_Writer
from architecture.transformer import AttnMIL6,clam
from modules import attmil,sdmil,rrty, WIKG, mean_max,mhim,AEM,ACMIL,transmil,OODML,fuzzy
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy
from dataloader import *
from utils1 import *
import torch.nn.functional as F
from copy import deepcopy


def get_arguments():
    parser = argparse.ArgumentParser('WSI classification training', add_help=False)
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--seed", type=int, default=4, help="set the random seed to ensure reproducibility" )
    
    parser.add_argument( "--mask_drop", type=float, default=0.5, help="number of query token" )
    parser.add_argument("--arch", type=str, default='mapmil', help="choice of architecture type")
    parser.add_argument("--numGroup", type=int, default='6', help="num of pseudo_bag")

    parser.add_argument('--high_medium_line', default=0.01, type=float, help='Dropout in the R-MSA')
    parser.add_argument('--medium_low_line', default=0.7, type=float, help='Dropout in the R-MSA')
    parser.add_argument("--diff_loss", type=str, default='diff', help="num of pseudo_bag")
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--cv_fold', default=1, type=int, help='Val-set ratio')
    parser.add_argument('--title', default='title', type=str, help='Val-set ratio')
    parser.add_argument('--teacher_init', default='none', type=str, help='Path to initial teacher model')
    parser.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none,fc,all]')
    parser.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization')
    parser.add_argument('--tea_type', default='none', type=str, help='[none,same]')
    
    parser.add_argument('--topk', default=0.05, type=float, help='number of warm-up epochs')
    parser.add_argument('--dataset_root', default='/data/xxx/TCGA', type=str, help='Dataset root path')
    parser.add_argument('--train_epoch', default=150, type=int, help='Number of total training epochs')
    parser.add_argument('--dataset', default='gastriccancer', type=str, help='name of dataset')
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--n_worker', default=8, type=int)
    parser.add_argument('--pretrain', default='ResNet50', type=str, help='feature extract network')
    parser.add_argument('--input_dim', default=1024, type=int, help='dimensions of input features')
    parser.add_argument('--D_inner', default=256, type=int, help='Embedded layer dimension')
    parser.add_argument('--weight_decay', default=0.00001, type=float, help=' weight decay')
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--min_lr', default=0, type=int)
    parser.add_argument('--warmup_epoch', default=0, type=int, help='number of warm-up epochs')
    
    parser.add_argument('--lamda_pse', default=1.0,type=float)
    parser.add_argument('--lamda_kl', default=1.0, type=float)
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    seed_torch(args.seed)
    if args.dataset.lower() in ['musk1', 'musk2', 'fox', 'tiger', 'elephant']:
        csv_path = os.path.join('./dataset_csv/', f'{args.dataset.capitalize()}.csv')
        p, l = get_benchmarks_label(csv_path)
        index = [i for i in range(len(p))]
        np.random.shuffle(index)
        p = p[index]
        l = l[index]
           
    if args.dataset.lower() == 'camelyon16':
        args.ckpt_dir = os.path.join('RESULT','CAMELYON16', args.pretrain,args.title) 
    elif args.dataset.lower() == 'gastriccancer':
        args.ckpt_dir = os.path.join('RESULT','GasticCancer_v2.0',args.pretrain,args.title)  
    elif args.dataset.lower() == 'tcga':
        args.ckpt_dir = os.path.join('RESULT', 'TCGA-LUNG',args.pretrain,args.title)  
    elif args.dataset.lower() in ['musk1','musk2','fox','tiger','elephant']:
        args.ckpt_dir = os.path.join('RESULT','BENCHMARKS', args.dataset.lower(),args.title)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    if args.dataset.lower() == 'gastriccancer':
        train_p, train_l, val_p, val_l, test_p, test_l = get_train_test_from_csv('./dataset_csv/GasticCancer_v2.0.csv',dataset=args.dataset)
    elif args.dataset.lower() == 'camelyon16':
        train_p, train_l, val_p, val_l, test_p, test_l = get_train_test_from_csv('./dataset_csv/camelyon16.csv',dataset=args.dataset)
    elif args.dataset.lower() == 'tcga':
        train_p, train_l, val_p, val_l, test_p, test_l = get_train_test_from_csv('./dataset_csv/tcga-lung.csv',dataset=args.dataset)
    elif args.dataset.lower() in ['musk1', 'musk2', 'fox', 'tiger', 'elephant']:
        train_p, train_l, test_p, test_l,val_p, val_l = get_kflod(args.cv_fold, p, l, args.val_ratio)
    
    acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]
    print('Dataset: ' + args.dataset)
    
    log_dir = os.path.join(args.ckpt_dir, 'train')  
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'{args.title}.txt')

    for k in range(0, args.cv_fold):

        print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l,log_file)

        print('Cross validation accuracy mean: %.6f, std %.6f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        print('Cross validation auc mean: %.6f, std %.6f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        print('Cross validation precision mean: %.6f, std %.6f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        print('Cross validation recall mean: %.6f, std %.6f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        print('Cross validation fscore mean: %.6f, std %.6f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))
  
def one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l,log_file):
    # ---> Initialization
    seed_torch(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric
   
    # ---> Loading data
    if args.dataset.lower() == 'camelyon16':
        train_set = C16Dataset(train_p,train_l,root=args.dataset_root,persistence=False,keep_same_psize=0,is_train=True)
        test_set = C16Dataset(test_p,test_l,root=args.dataset_root,persistence=False,keep_same_psize=0)

    elif args.dataset.lower() == 'tcga':
        train_set = TCGADataset(train_p,train_l,args.dataset_root,persistence=False,keep_same_psize=0,is_train=True)
        val_set = TCGADataset(val_p,val_l,args.dataset_root,persistence=False,keep_same_psize=0)
        test_set = TCGADataset(test_p,test_l,args.dataset_root,persistence=False,keep_same_psize=0)
        
    elif args.dataset.lower() == 'gastriccancer':
        train_set = GastricCancer_v2(train_p,train_l,root=args.dataset_root,persistence=False,keep_same_psize=0,is_train=True)
        test_set = GastricCancer_v2(test_p,test_l,root=args.dataset_root,persistence=False,keep_same_psize=0)


    elif args.dataset.lower() == 'musk1' or args.dataset.lower() == 'musk2' or args.dataset.lower() == 'fox' or args.dataset.lower() == 'tiger' or args.dataset.lower() == 'elephant':
        train_set = Benchmarks(train_p[k],train_l[k],root=args.dataset_root,dataset_name=args.dataset.lower())
        test_set = Benchmarks(test_p[k],test_l[k],root=args.dataset_root,dataset_name=args.dataset.lower())


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.n_worker, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,num_workers=args.n_worker, pin_memory=False, drop_last=False)
    if args.dataset.lower() == 'tcga':
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,num_workers=args.n_worker, pin_memory=False, drop_last=False)
    else:
        val_loader = test_loader
                
    if not args.teacher_init.endswith('.pth'):
        _str = 'checkpoint-best-{}-auc.pth'.format(k)
        _teacher_init = os.path.join(args.teacher_init,_str)
    else:
        _teacher_init =args.teacher_init

    # define network
    if args.arch == 'mapmil':
        milnet = AttnMIL6(args).to(device)
    elif  args.arch == 'gattmil':
        milnet = attmil.AttentionGated(args.input_dim,dropout=0.25).to(device)
    elif args.arch == 'abmil':
        milnet = attmil.DAttention(args).to(device)
    elif args.arch == 'rrtmil':
        if args.dataset.lower() == 'tcga':
                model_params = {
                'input_dim': args.input_dim,
                'n_classes': args.n_classes,
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
                'crmsa_mlp': False,
                'crmsa_heads':8,
            }
        else:
            model_params = {
                'input_dim': args.input_dim,
                'n_classes': args.n_classes,
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
                'epeg_k': 15,
                'epeg_2d': False,
                'epeg_bias': True,
                'epeg_type': 'attn',
                'region_attn': 'native',
                'peg_1d': False,
                'cr_msa': True,
                'crmsa_k': 1,
                'all_shortcut': True,
                'crmsa_mlp': False,
                'crmsa_heads':8,
            }
        milnet = rrty.RRTMIL(**model_params).to(device)
    elif args.arch == 'sdmil':
        if args.dataset.lower() == 'tcga':
                model_params = {
                'input_dim': args.input_dim,
                'n_classes': args.n_classes,
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
        else:
            model_params = {
                'input_dim': args.input_dim,
                'n_classes': args.n_classes,
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
                'epeg_k': 15,
                'epeg_2d': False,
                'epeg_bias': True,
                'epeg_type': 'attn',
                'region_attn': 'native',
                'peg_1d': False,
                'cr_msa': True,
                'crmsa_k': 1,
                'all_shortcut': True,
                'crmsa_mlp':False,
                'crmsa_heads':1,}
        milnet = sdmil.SDMIL(**model_params).to(device)
    
    elif args.arch == 'maxmil':   
        milnet = mean_max.MaxMIL(args,args.n_classes,act='relu').to(device)
    elif args.arch == 'meanmil':  
        milnet = mean_max.MeanMIL(args,args.n_classes,act='relu').to(device)
    elif  args.arch == 'wikg':  
        milnet = WIKG.WiKG(args, dim_hidden=512, topk=6, n_classes=args.n_classes, agg_type='bi-interaction', dropout=0.3, pool='mean').to(device) 
    elif args.arch == 'clam_sb':
        milnet = clam.CLAM_SB(input_dim=args.input_dim,n_classes=args.n_classes,dropout=0.25,act='relu').to(device)
    elif args.arch == 'clam_mb':
        milnet = clam.CLAM_MB(input_dim=args.input_dim,n_classes=args.n_classes,dropout=0.25,act='relu').to(device)
    elif args.arch == 'transmil': 
        milnet = transmil.TransMIL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=0.25,act='relu').to(device)
    
    elif args.arch == 'mhimmil': 
        
        mrh_sche = cosine_scheduler(0.01,0.,epochs=args.train_epoch,niter_per_ep=len(train_loader))
        if args.dataset.lower() == 'tcga':
                model_params = {
                'baseline': 'attn',
                'dropout': 0.25,
                'mask_ratio' : 0.7,
                'n_classes': args.n_classes,
                'temp_t': 0.1,
                'act': 'relu',
                'head': 8,
                'msa_fusion': 'vote',
                'mask_ratio_h': 0.02,
                'mask_ratio_hr': 1.0,
                'mask_ratio_l': 0.2,
                'mrh_sche': mrh_sche,
                'da_act': 'relu',
                'attn_layer': 0,
                'feat_dim': args.input_dim}
        else:
            model_params = {
                'baseline': 'attn',
                'dropout': 0.25,
                'mask_ratio' : 0.5,
                'n_classes': args.n_classes,
                'temp_t': 0.1,
                'act': 'relu',
                'head': 8,
                'msa_fusion': 'vote',
                'mask_ratio_h': 0.01,
                'mask_ratio_hr': 0.5,
                'mask_ratio_l': 0.,
                'mrh_sche': mrh_sche,
                'da_act': 'relu',
                'attn_layer': 0,
                'feat_dim': args.input_dim}
        
        milnet = mhim.MHIM(**model_params).to(device)
    elif args.arch == 'pure':
        milnet = mhim.MHIM(select_mask=False,n_classes=args.n_classes,act='relu',head=8,da_act='relu',baseline='attn',feat_dim=args.input_dim).to(device)

    elif args.arch =='ACMIL':
        milnet = ACMIL.ACMIL(input_dim=args.input_dim, n_class=args.n_classes).to(device)  

    elif args.arch =='AEM':
        milnet = AEM.ABMIL(input_dim=args.input_dim, n_class=args.n_classes).to(device)   

    elif args.arch == 'mkmil':
        from modules import MKMIL,radam,lookahead
        milnet = MKMIL.MKMIL(n_classes=args.n_classes, dropout=0.5, act='relu', n_features=args.input_dim, layer=2, rate=10, type="AFWMamba").to(device)
        
    elif  args.arch == 'fuzzymil':
        model_params = {
            'input_dim': args.input_dim,
            'n_classes': args.n_classes,
            'dropout': 0.25,
            'act': 'relu',
            'region_num': 8,
            'pos': 'none',
            'pos_pos': 0,
            'pool': 'attn',
            'peg_k': 7,
            'drop_path': 0.,
            'n_layers':2,
            'n_heads': 8,
            'attn': 'rmsa',
            'da_act': 'tanh',
            'ffn': False,
            'mlp_ratio': 4.,
            'trans_dim':64,
            'epeg': True,
            'min_region_num': 0,
            'qkv_bias': True,
            'epeg_k': 15,
            'epeg_2d': False,
            'epeg_bias': True,
            'epeg_type': 'attn',
            'region_attn': 'native',
            'peg_1d': False,
            'cr_msa':True,
            'crmsa_k': 1,
            'all_shortcut':True,
            'crmsa_mlp':False,
            'crmsa_heads':8,
         }
        milnet = fuzzy.FuzzyMIL(**model_params).to(device)

    else:
        raise KeyError
    
    if args.arch == 'mhimmil': 
        if args.init_stu_type != 'none':
            print('######### Model Initializing.....')
            pre_dict = torch.load(_teacher_init)
            new_state_dict ={}
            if args.init_stu_type == 'fc':
            # only patch_to_emb
                for _k,v in pre_dict.items():
                    _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                    new_state_dict[_k]=v
                info = milnet.patch_to_emb.load_state_dict(new_state_dict,strict=False)
            else:
            # init all
                info = milnet.load_state_dict(pre_dict,strict=False)
            print(info)
        
    # teacher model
    if args.arch == 'mhimmil':
        milnet_tea = deepcopy(milnet)
        if not args.no_tea_init and args.tea_type != 'same':
            print('######### Teacher Initializing.....')
            try:
                pre_dict = torch.load(_teacher_init)
                info = milnet_tea.load_state_dict(pre_dict,strict=False)
                print(info)
            except:
                print('########## Init Error')
        if args.tea_type == 'same':
            milnet_tea = milnet
    else:
        milnet_tea = None

    if args.dataset.lower() == 'tcga':
            mm_sche = cosine_scheduler(0.9999,1.,epochs=args.train_epoch,niter_per_ep=len(train_loader),start_warmup_value=1.)
    else:
        mm_sche=None
    
    criterion = nn.CrossEntropyLoss()
    # define optimizer, lr not important at this point
   
    if args.arch == 'mkmil':
        optimizer = radam.RAdam(milnet.parameters(),lr=0.0002, weight_decay=0.00001)
        optimizer0 = lookahead.Lookahead(optimizer)
    elif args.arch == 'fuzzymil':
        optimizer0 = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()), lr=0.0002, weight_decay=0.00001)
    else:
        optimizer0 = torch.optim.AdamW(filter(lambda p: p.requires_grad, milnet.parameters()), lr=0.001, weight_decay=0.00001)
    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_pre':0,'val_rec':0,'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_pre':0,'test_rec':0, 'test_f1':0}

    for epoch in range(args.train_epoch):
        train_loss= train_one_epoch(milnet, milnet_tea, criterion, train_loader, optimizer0, device, epoch, args, mm_sche)
        val_auc, val_acc, val_pre,val_rec,val_f1, val_loss = evaluate(milnet,milnet_tea, criterion, val_loader, device, args, 'Val')
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.train_epoch} | Acc: {val_acc:.6f}, AUC: {val_auc:.6f}, F1: {val_f1:.6f}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}\n")
        
        if   val_auc >  best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['val_pre'] = val_pre
            best_state['val_rec'] = val_rec
            save_model(conf=args, k=k,model=milnet, optimizer=optimizer0, epoch=epoch, base_ckpt_dir=args.ckpt_dir,is_best_auc=True)
            
    save_model(conf=args, k=k, model=milnet, optimizer=optimizer0, epoch=epoch,base_ckpt_dir=args.ckpt_dir, is_last=True)
    print('best_state:',best_state)
    
    if args.dataset.lower() == 'tcga':
        best_auc_model_path = os.path.join(args.ckpt_dir, f'checkpoint-best-{k}-auc.pth')  
        checkpoint = torch.load(best_auc_model_path)
        milnet.load_state_dict(checkpoint['model'])
        test_auc, test_acc, test_pre,test_rec,test_f1, test_loss = evaluate(milnet,milnet_tea, criterion, test_loader, device, args, 'Val')
        print('\n best_auc accuracy: %.6f ,best_auc auc: %.6f,best_auc precision: %.6f,best_auc recall: %.6f,best_auc fscore: %.6f' % (test_acc,test_auc,test_pre,test_rec,test_f1))
        
    acs.append(best_state['val_acc'])
    pre.append(best_state['val_pre'])
    rec.append(best_state['val_rec'])
    fs.append(best_state['val_f1'])
    auc.append(best_state['val_auc'])
    
    return [acs,pre,rec,fs,auc,te_auc,te_fs]


def train_one_epoch(milnet, milnet_tea, criterion, data_loader, optimizer0, device, epoch, args, mm_sche=None):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """
    # Set the network to training mode
    milnet.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    
    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image_patches = data[0].to(device, dtype=torch.float32)
        coords = data[2].to(device)
        labels = data[1].to(device)
        
        adjust_learning_rate(optimizer0, epoch + data_it/len(data_loader), args)
        if milnet_tea is not None:
            milnet_tea.train()

        if args.arch=='mapmil':
           
            slide_preds, attn, bag_feature = milnet(image_patches, use_attention_mask=True,return_bag_feature=True,mask_drop=args.mask_drop)
            loss_cls = criterion(slide_preds, labels)
            
            _,n=attn.squeeze(0).shape  
            
            if n < args.numGroup * 3:
                pse_loss = torch.tensor(0.0).to(device)
                diff_loss = torch.tensor(0.0).to(device)
                loss = loss_cls + pse_loss + diff_loss # 
                
            else:
                
                image_patches=image_patches.squeeze(0)
                _, indices = torch.topk(attn, n, dim=-1)
                #indices = indices.squeeze().cpu().numpy()
                indices= indices.flatten().cpu().numpy()
            
                n=len(indices)
                
                ####################################################################################
                '''
                np.random.shuffle(indices)

                indices_groups = np.array_split(indices, args.numGroup)

                pseudo_bag = []
                for group in indices_groups:
                    selected_instances = image_patches[group]  
                    pseudo_bag.append(selected_instances)
                '''
                ####################################################################################
            
                split_1 = int(n * args.high_medium_line)     # 
                split_2 = int(n * args.medium_low_line)      # 
                split_3 = n 
                
                top_1_percent = indices[:split_1]         # 
                middle_1_to_50_percent = indices[split_1:split_2]  # 
                bottom_50_to_100_percent = indices[split_2:split_3]  #
                
                '''
                #np.random.shuffle(top_1_percent)
                #np.random.shuffle(middle_1_to_50_percent)
                #np.random.shuffle(bottom_50_to_100_percent)
                '''
                
                #second_third_integrate
                #bottom_50_to_100_percent = indices[split_1:split_3]
                #keep_second
                #bottom_50_to_100_percent = indices[split_1:split_2]
                #keep_third
                #bottom_50_to_100_percent = indices[split_2:split_3]  # 

                top_1_percent_list = top_1_percent.tolist()
                middle_1_to_50_percent_list = middle_1_to_50_percent.tolist()
                #middle_1_to_50_percent_list = middle_1_to_50_percent.tolist()[::-1] 
                bottom_50_to_100_percent_list = bottom_50_to_100_percent.tolist()
                #bottom_50_to_100_percent_list = bottom_50_to_100_percent.tolist()[::-1]

                top_1_percent_list_numGroup = np.array_split(top_1_percent_list, args.numGroup)
                middle_1_to_50_percent_list_numGroup = np.array_split(middle_1_to_50_percent_list, args.numGroup)
                bottom_50_to_100_percent_list_numGroup = np.array_split(bottom_50_to_100_percent_list, args.numGroup)
                
                pseudo_index_batches = []
                for top, middle, bottom in zip(top_1_percent_list_numGroup, middle_1_to_50_percent_list_numGroup, bottom_50_to_100_percent_list_numGroup):
                
                #for top, bottom in zip(top_1_percent_list_numGroup, bottom_50_to_100_percent_list_numGroup):
                    #pseudo_batch = np.concatenate([top, bottom])
                    
                #for top, middle in zip(top_1_percent_list_numGroup, middle_1_to_50_percent_list_numGroup):
                    #pseudo_batch = np.concatenate([top, middle])
                    
                    pseudo_batch = np.concatenate([top, middle, bottom])
                    pseudo_index_batches.append(pseudo_batch)
                    #print('pseudo_batch.shape:',pseudo_batch.shape)
                #print('image_patches.shape:',image_patches.shape)
                pseudo_bag=[]
                
                for i, pseudo_batch in enumerate(pseudo_index_batches):
                    selected_instances = image_patches[pseudo_batch]  
                    pseudo_bag.append(selected_instances) 
                    
                label_groups = [labels for _ in range(args.numGroup)]
                pseudo_bag_total_loss=0.0
                
                pseudo_bag_fea_list=[]
                pseudo_bag_logit_list=[]
                for group_idx, (group_bag, group_label) in enumerate(zip(pseudo_bag, label_groups)):
                    group_bag=group_bag.unsqueeze(0)
                    slide_preds, attn, pseudo_bag_fea = milnet(group_bag, use_attention_mask=False,pseudo_bag=True)
                
                    pseudo_bag_fea_list.append(pseudo_bag_fea)
                    pseudo_bag_logit_list.append(slide_preds)
                    pseudo_bag_logit_loss = criterion(slide_preds.view(1, -1), group_label)
                    pseudo_bag_total_loss += pseudo_bag_logit_loss
                
                pse_loss = pseudo_bag_total_loss / args.numGroup
                pseudo_bag_fea_sum = torch.cat(pseudo_bag_fea_list, dim=0) 
                pseudo_bag_logit_sum = torch.cat(pseudo_bag_logit_list, dim=0) 
                #print('pseudo_bag_fea_sum.shape:',pseudo_bag_fea_sum.shape)

                pseudo_bag_num,_=pseudo_bag_fea_sum.shape  
                diff_loss = torch.tensor(0).to(device, dtype=torch.float)
                for i in range(pseudo_bag_num):
                    for j in range(i + 1, pseudo_bag_num):
                        
                        if args.diff_loss=='cosine':
                            diff_loss += 1- torch.cosine_similarity(pseudo_bag_fea_sum[i], pseudo_bag_fea_sum[j], dim=-1).mean() / (pseudo_bag_num * (pseudo_bag_num - 1) / 2)
                        elif args.diff_loss=='MSE':
                            diff_loss += F.mse_loss(pseudo_bag_fea_sum[ i], pseudo_bag_fea_sum[ j]).mean() / (pseudo_bag_num * (pseudo_bag_num - 1) / 2)    
                        elif args.diff_loss=='NONE':
                            diff_loss += torch.tensor(0).to(device, dtype=torch.float) 
                        elif args.diff_loss=='KL_loss':
                            if args.dataset.lower() in ['musk1', 'musk2', 'fox', 'tiger', 'elephant']:
                                diff_loss += F.kl_div(F.log_softmax(pseudo_bag_logit_sum[i].unsqueeze(0), dim=-1), F.log_softmax(pseudo_bag_logit_sum[j].unsqueeze(0), dim=-1), reduction='batchmean', log_target=True) 
                            else:
                                diff_loss += F.kl_div(F.log_softmax(pseudo_bag_logit_sum[i].unsqueeze(0), dim=-1), F.softmax(pseudo_bag_logit_sum[j].unsqueeze(0), dim=-1), reduction='batchmean')
                            
                    if args.diff_loss=='MSE*':
                        diff_loss += F.mse_loss(pseudo_bag_fea_sum[i].unsqueeze(0), pseudo_bag_fea_sum.mean(dim=0).unsqueeze(0))
                    
                loss = args.lamda_kl*diff_loss + loss_cls + args.lamda_pse*pse_loss 
        
        elif args.arch in ('clam_sb','clam_mb'):
            train_logits,cls_loss,_ = milnet(image_patches,labels,criterion)
            loss1 = criterion(train_logits, labels)
            if args.arch=='clam_sb':
                loss= 0.7*loss1 + 0.3*cls_loss
            elif args.arch=='clam_mb':
                loss= 0.5*loss1 + 0.5*cls_loss
            else:
                raise KeyError
        elif args.arch == 'mhimmil':
            if milnet_tea is not None:
                cls_tea,attn = milnet_tea.forward_teacher(image_patches,return_attn=True)
            else:
                attn,cls_tea = None,None

            preds, cls_loss,patch_num,keep_num = milnet(image_patches,attn,cls_tea,i=epoch*len(data_loader)+data_it)
            loss1 = criterion(preds, labels)
            if args.dataset.lower() == 'tcga':
                loss =  loss1 +  cls_loss*0.5
            else:
                loss =  loss1 +  cls_loss*0.1
                
        elif args.arch == 'pure':
            preds, cls_loss,patch_num,keep_num = milnet.pure(image_patches)
            loss = criterion(preds, labels)
              
        elif args.arch == 'ACMIL':
            sub_preds,train_logits,attn  = milnet(image_patches)
            loss1 = criterion(sub_preds, labels.repeat_interleave(5))
            logit_loss = criterion(train_logits,labels)
            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attn = torch.softmax(attn, dim=-1)

            for i in range(5):
                for j in range(i + 1, 5):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (5 * (5 - 1) / 2)
            loss = diff_loss + loss1 + logit_loss

        elif args.arch == 'AEM':
            train_logits,attn  = milnet(image_patches)
            loss1 = criterion(train_logits,labels)
            div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1))
            loss = 0.1 * div_loss + loss1      
        else :
            preds = milnet(image_patches)
            loss = criterion(preds, labels)
            
        optimizer0.zero_grad()
        # Backpropagate error and update parameters
        loss.backward()

        optimizer0.step()

        if args.arch == 'mhimmil':
            if mm_sche is not None:
                mm = mm_sche[epoch*len(data_loader)+data_it]
            else:
                mm = 0.9999
            if milnet_tea is not None:
                if args.tea_type == 'same':
                    pass
                else:
                    ema_update(milnet,milnet_tea,mm)
            
        if args.arch == 'mapmil': 
            metric_logger.update(lr=optimizer0.param_groups[0]['lr'])
            metric_logger.update(all_loss=loss.item())
            metric_logger.update(the_mask_branch=loss_cls.item())
            metric_logger.update(the_pseudo_branch=pse_loss.item())
            metric_logger.update(diff_loss=diff_loss.item())
        else:
            metric_logger.update(lr=optimizer0.param_groups[0]['lr'])
            metric_logger.update(all_loss=loss.item())
            
    return loss

# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(net, milnet_tea, criterion, data_loader, device, args, header):
    # Set the network to evaluation mode
    net.eval()
    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter="  ")
    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data[0].to(device, dtype=torch.float32)
        labels = data[1].to(device)
        coords = data[2].to(device)
        if args.arch=='mapmil' :
            preds, attn = net(image_patches, use_attention_mask=False,pseudo_bag=False)
        elif args.arch in ('mhimmil','pure'):
            preds = net.forward_test(image_patches)    
        elif args.arch in ('clam_sb','clam_mb'):
            preds = net(image_patches,instance_eval=False)   
        elif args.arch =='AEM':
            preds, attn  = net(image_patches)
        elif args.arch =='ACMIL':
            sub_preds, preds, attn  = net(image_patches)         
        else:
            preds = net(image_patches)
            
        loss = criterion(preds, labels)
        pred = torch.softmax(preds, dim=-1)[:,1]
        metric_logger.update(loss=loss.item())
        
        y_pred.append(pred)
        y_true.append(labels)
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy() 
    y_true = torch.cat(y_true, dim=0).cpu().numpy() 
    
    accuracy, auc_value, precision, recall, fscore = five_scores(y_true, y_pred)
    print('\n test accuracy: %.6f ,test auc: %.6f,test precision: %.6f,test recall: %.6f,test fscore: %.6f' % (accuracy,auc_value,precision,recall,fscore))
    return auc_value, accuracy, precision, recall,fscore, metric_logger.loss.global_avg


if __name__ == '__main__':
    main()

#Camelyon16
#python main.py --seed 4 --arch mapmil --dataset_root=./Camelyon16/ResNet50  --dataset=camelyon16 --pretrain ResNet50 --input_dim=1024 --D_inner=256 --n_classes 2 --train_epoch 150 --topk=0.05  --high_medium_line 0.05 --mask_drop 0.05 --medium_low_line 0.4  --title mapmil  --numGroup=4 --diff_loss=MSE*
#python main.py --seed 4 --arch mapmil  --dataset_root=./Camelyon16/CtransPath  --dataset=camelyon16 --pretrain CtransPath --input_dim=768 --D_inner=512 --n_classes 2 --train_epoch 150 --topk=0.05  --high_medium_line 0.01 --mask_drop 0.95 --medium_low_line 0.5  --title mapmil  --numGroup=2 --diff_loss=MSE*

#TCGA-LUNG
#python main.py --seed 4   --arch mapmil   --dataset_root=./TCGA-LUNG/ResNet50  --dataset=tcga --pretrain ResNet50  --input_dim=1024 --D_inner=256 --n_classes 2 --train_epoch 150 --topk=0.05  --high_medium_line 0.05 --mask_drop 0.25 --medium_low_line 0.3 --title mapmil  --numGroup=6 --diff_loss=KL_loss --val_ratio=0.13
#python main.py --seed 4   --arch mapmil   --dataset_root=./TCGA-LUNG/CtransPath  --dataset=tcga --pretrain CtransPath --input_dim=768 --D_inner=512 --n_classes 2 --train_epoch 150 --topk=0.05   --high_medium_line 0.01 --mask_drop 0.25 --medium_low_line 0.5 --title mapmil  --numGroup=2 --diff_loss=KL_loss --val_ratio=0.13

#MUSK1
#python main.py --seed 2021 --arch mapmil  --dataset_root=./dataset_csv  --dataset=Musk1 --pretrain Traditonal --input_dim=166 --D_inner=256 --n_classes 2 --train_epoch 70 --topk=0.01 --high_medium_line 0.3 --mask_drop 0.2 --medium_low_line 0.6 --cv_fold=5 --title mapmil  --numGroup=4 --diff_loss=KL_loss 

#MUSK2
#python main.py --seed 2021 --arch mapmil  --dataset_root=./dataset_csv  --dataset=Musk2 --pretrain Traditonal  --input_dim=166 --D_inner=256 --n_classes 2 --train_epoch 70 --topk=0.01 --high_medium_line 0.3 --mask_drop 0.3 --medium_low_line 0.6 --cv_fold=5 --title mapmil  --numGroup=4 --diff_loss=KL_loss 

#Elephant
#python main.py --seed 2021 --arch mapmil  --dataset_root=./dataset_csv  --dataset=Elephant --pretrain Traditonal  --input_dim=230 --D_inner=256 --n_classes 2 --train_epoch 70  --topk=0.01 --high_medium_line 0.2 --mask_drop 0.1 --medium_low_line 0.6 --cv_fold=5 --title mapmil  --numGroup=4 --diff_loss=KL_loss 




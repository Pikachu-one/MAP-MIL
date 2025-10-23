
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
    
    log_dir = os.path.join(args.ckpt_dir, 'test')  
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
                
    # define network
    if args.arch == 'mapmil':
        milnet = AttnMIL6(args).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # define optimizer, lr not important at this point
    
    if args.dataset.lower() == 'camelyon16':
        if args.pretrain=='ResNet50':
            dir='./Result/CAMELYON16/ResNet50/mapmil/weight'  
        elif   args.pretrain=='CtransPath':
            dir='./Result/CAMELYON16/CtransPath/mapmil/weight'  
    elif args.dataset.lower() == 'tcga':
        if args.pretrain=='ResNet50':
            dir='./Result/TCGA-LUNG/ResNet50/mapmil/weight' 
        elif   args.pretrain=='CtransPath':
            dir='./Result/TCGA-LUNG/CtransPath/mapmil/weight' 
    elif args.dataset.lower() == 'musk1':
        dir='./Result/BENCHMARKS/musk1/mapmil/weight'
    elif args.dataset.lower() == 'musk2':
        dir='./Result/BENCHMARKS/musk2/mapmil/weight'
    elif args.dataset.lower() == 'elephant':
        dir='./Result/BENCHMARKS/elephant/mapmil/weight'    
   
    best_auc_model_path = os.path.join(dir, f'checkpoint-best-{k}-auc.pth')  
    checkpoint = torch.load(best_auc_model_path)
    milnet.load_state_dict(checkpoint['model'])    
    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_pre':0,'val_rec':0,'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_pre':0,'test_rec':0, 'test_f1':0}
         
    milnet_tea = None
    val_auc, val_acc, val_pre,val_rec,val_f1, val_loss = evaluate(milnet,milnet_tea, criterion, test_loader, device, args, 'Val')
    print('\n val accuracy: %.6f ,val auc: %.6f,val precision: %.6f,val recall: %.6f,val fscore: %.6f' % (val_acc,val_auc,val_pre,val_rec,val_f1))
    with open(log_file, 'a') as f:
        f.write(f" val Acc: {val_acc:.6f}, val AUC: {val_auc:.6f}, val Pre:  {val_pre:.6f}, val Rec:  {val_rec:.6f}, val F1: {val_f1:.6f}\n")
    if   val_auc >  best_state['val_auc']:
        best_state['val_auc'] = val_auc
        best_state['val_acc'] = val_acc
        best_state['val_f1'] = val_f1
        best_state['val_pre'] = val_pre
        best_state['val_rec'] = val_rec        
    
        
    acs.append(best_state['val_acc'])
    pre.append(best_state['val_pre'])
    rec.append(best_state['val_rec'])
    fs.append(best_state['val_f1'])
    auc.append(best_state['val_auc'])
    
    return [acs,pre,rec,fs,auc,te_auc,te_fs]


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
#python test.py --seed 4 --arch mapmil --dataset_root=./Camelyon16/ResNet50  --dataset=camelyon16 --pretrain ResNet50 --input_dim=1024 --D_inner=256 --n_classes 2   --title mapmil  
#python test.py --seed 4 --arch mapmil  --dataset_root=./Camelyon16/CtransPath  --dataset=camelyon16 --pretrain CtransPath --input_dim=768 --D_inner=512 --n_classes 2  --title mapmil 

#TCGA-LUNG
#python test.py --seed 4   --arch mapmil   --dataset_root=./TCGA-NSCLC_R50  --dataset=tcga --pretrain ResNet50  --input_dim=1024 --D_inner=256 --n_classes 2  --title mapmil --val_ratio=0.13
#python test.py --seed 4   --arch mapmil   --dataset_root=./TCGA_ctranspath_Extracted_feature  --dataset=tcga --pretrain CtransPath --input_dim=768 --D_inner=512 --n_classes 2 --title mapmil  --val_ratio=0.13

#MUSK1
#python test.py --seed 2021 --arch mapmil  --dataset_root=./dataset_csv  --dataset=Musk1 --pretrain Traditonal --input_dim=166 --D_inner=256 --n_classes 2 --cv_fold=5 --title mapmil 

#MUSK2
#python test.py --seed 2021 --arch mapmil  --dataset_root=./dataset_csv  --dataset=Musk2 --pretrain Traditonal  --input_dim=166 --D_inner=256 --n_classes 2  --cv_fold=5 --title mapmil 

#Elephant
#python test.py --seed 2021 --arch mapmil  --dataset_root=./dataset_csv  --dataset=Elephant --pretrain Traditonal  --input_dim=230 --D_inner=256 --n_classes 2  --cv_fold=5 --title mapmil  




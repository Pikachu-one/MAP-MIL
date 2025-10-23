

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import cv2
import random
import os
from wsi_core.WholeSlideImage import WholeSlideImage
import argparse, os
from architecture.transformer import AttnMIL6
def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

parser = argparse.ArgumentParser(description='MIL Training Script')
parser.add_argument('--f_id', default='1', type=str)
parser.add_argument('--datasets', default='camelyon16', type=str)
parser.add_argument('--model_vis', default='mapmil', type=str)
parser.add_argument("--seed", type=int, default=4, help="set the random seed to ensure reproducibility" )

parser.add_argument("--input_dim", default=1024, type=int)
parser.add_argument('--D_inner', default=256, type=int)
parser.add_argument('--n_classes', default=2, type=int)
args = parser.parse_args()

args = parser.parse_args()
seed_torch(args.seed)
f_id = args.f_id

tif_path = "/CAMELYON16/"+f_id+".tif"
json_path = "/camelyon16_train_json/"+f_id+".json"
h5_path = "/h5_files/"+f_id+".h5"
try:
    with open(json_path, 'r') as f:
        json_data = json.load(f)
except:
    json_data = None
    print('Normal slide')

slide = WholeSlideImage(tif_path)
patch = h5py.File(h5_path,"r")
features = torch.Tensor(patch['features'])
coords = np.array(patch['coords'])              
    
model = AttnMIL6(args)
model.requires_grad_(False)

#weight path
mask_cpt = torch.load('XXXX')
model.load_state_dict(mask_cpt['model'],strict=True)   

_,A,_= model(features)   

print('A.shape:',A.shape)
print('A:',A)
probs = torch.softmax(A, dim=-1)
probs = probs.view(-1,1).cpu().numpy()  
folder_path = os.path.join(".vis/Result", f_id)
os.makedirs(folder_path, exist_ok=True) 
heatmap = slide.visHeatmap(scores= probs*probs.size*probs.size*200, coords=coords, patch_size=(512, 512), segment=False, cmap='jet')
file_path = os.path.join(folder_path, f"{args.model_vis}_vis.png")
heatmap.save(file_path)









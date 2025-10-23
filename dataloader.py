import os
import csv
import torch
import random
import numpy as np
import pandas as pd

from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import h5py
def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

import pandas as pd
import numpy as np
from collections import Counter

def get_benchmarks_label(csv_file):
    try:
       
        df = pd.read_csv(csv_file, header=None, usecols=[0, 1])
        df.columns = ['Bag_Label', 'Bag_ID']
        
    except Exception as e:
        print(f"读取 CSV 文件时出错，请检查 {csv_file} 是否存在且格式正确: {e}")
        return np.array([]), np.array([])
    
    unique_bags = df.drop_duplicates(subset=['Bag_ID']).reset_index(drop=True)
    
    patients_array = unique_bags['Bag_ID'].values
    labels_array = unique_bags['Bag_Label'].values
    
    # 5. 打印统计信息 (Bag 级别的统计)
    a = Counter(labels_array)
    print("Unique Bag ID length: {}".format(len(patients_array)))
    print("Unique Bag Label length: {}".format(len(labels_array)))
    print("All Bag Label counts: {}".format(dict(a)))
    
    return patients_array, labels_array

    
def get_train_test_from_csv(csv_path,dataset):
    
    df = pd.read_csv(csv_path)

    train_df = df[df['train'].notna()]
    train_p = train_df['train'].astype(str).values.tolist()
    train_l = train_df['train_label'].astype(str).values.tolist()

    val_p = []
    val_l = []
    if 'val' in df.columns and 'val_label' in df.columns:
        val_df = df[df['val'].notna()]
        val_p = val_df['val'].astype(str).values.tolist()
        val_l = val_df['val_label'].astype(str).values.tolist()
    
    if dataset.lower() == 'gastriccancer' :
        test_df = df[df['test'].notna()]
        test_p = test_df['test'].astype(int).astype(str).values.tolist()
        test_l = test_df['test_label'].astype(int).astype(str).values.tolist()
    elif dataset.lower() == 'camelyon16':
        test_df = df[df['test'].notna()]
        test_p = test_df['test'].astype(str).values.tolist()
        test_l = test_df['test_label'].astype(int).astype(str).values.tolist()
    else:
        test_df = df[df['test'].notna()]
        test_p = test_df['test'].astype(str).values.tolist()
        test_l = test_df['test_label'].astype(str).values.tolist()

    train_p = np.array(train_p, dtype=str)
    train_l = np.array(train_l, dtype=str)
    
    test_p = np.array(test_p, dtype=str)
    test_l = np.array(test_l, dtype=str)  
    
    val_p = np.array(val_p, dtype=str)
    val_l = np.array(val_l, dtype=str)     
        
    return train_p, train_l, val_p, val_l, test_p, test_l


def data_split(full_list, ratio, shuffle=True,label=None,label_balance_val=True):
    """
    dataset split: split the full_list randomly into two sublist (val-set and train-set) based on the ratio
    :param full_list: 
    :param ratio:     
    :param shuffle:  
    """
    # select the val-set based on the label ratio
    if label_balance_val and label is not None:
        _label = label[full_list]
        _label_uni = np.unique(_label)
        sublist_1 = []
        sublist_2 = []
        for _l in _label_uni:
            _list = full_list[_label == _l]
            n_total = len(_list)
            offset = int(n_total * ratio)
            if shuffle:
                random.shuffle(_list)
            sublist_1.extend(_list[:offset])
            sublist_2.extend(_list[offset:])
        return sublist_1, sublist_2
    else:
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        val_set = full_list[:offset]
        train_set = full_list[offset:]
        return val_set, train_set


def get_kflod(k, patients_array, labels_array,val_ratio=False,label_balance_val=True):
    if k > 1:
        skf = StratifiedKFold(n_splits=k)
    else:
        raise NotImplementedError
    train_patients_list = []
    train_labels_list = []
    test_patients_list = []
    test_labels_list = []
    val_patients_list = []
    val_labels_list = []
    for train_index, test_index in skf.split(patients_array, labels_array):
        if val_ratio != 0.:
            val_index,train_index = data_split(train_index,val_ratio,True,labels_array,label_balance_val)
            x_val, y_val = patients_array[val_index], labels_array[val_index]
        else:
            x_val, y_val = [],[]
        x_train, x_test = patients_array[train_index], patients_array[test_index]
        y_train, y_test = labels_array[train_index], labels_array[test_index]

        train_patients_list.append(x_train)
        train_labels_list.append(y_train)
        test_patients_list.append(x_test)
        test_labels_list.append(y_test)
        val_patients_list.append(x_val)
        val_labels_list.append(y_val)
        
    # print("get_kflod.type:{}".format(type(np.array(train_patients_list))))
    return np.array(train_patients_list,dtype=object), np.array(train_labels_list,dtype=object), np.array(test_patients_list,dtype=object), np.array(test_labels_list,dtype=object),np.array(val_patients_list,dtype=object), np.array(val_labels_list,dtype=object)

def get_tcga_parser(root,cls_name,mini=False):
        x = []
        y = []

        for idx,_cls in enumerate(cls_name):
            _dir = 'mini_pt' if mini else 'pt_files'
            _files = os.listdir(os.path.join(root,_cls,'features',_dir))
            _files = [os.path.join(os.path.join(root,_cls,'features',_dir,_files[i])) for i in range(len(_files))]
            x.extend(_files)
            y.extend([idx for i in range(len(_files))])
            
        return np.array(x).flatten(),np.array(y).flatten()




class GastricCancer_v2(Dataset):
    def __init__(self, file_name,file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(GastricCancer_v2, self).__init__()
        self.patient_name = file_name
        self.patient_label = file_label
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.slide_label = []
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    self.slide_name.append(torch.load(os.path.join(self.root,'pt_files',_sides[_idx])))
                else:
                    self.slide_name.append(_sides[_idx])
                self.slide_label.append(self.patient_label[i])
        self.slide_label = [ 0 if _l == '1' else 1 for _l in self.slide_label]
        print('self.slide_name:',self.slide_name)
    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        label = self.slide_label[idx]

        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path))
            '''
            h5_file_path = os.path.join(self.root, 'h5_files',file_path.replace(".pt", ".h5"))  
            with h5py.File(h5_file_path,'r') as hdf5_file:
                coord = hdf5_file['coords'][:]
            '''
            coord=0
        return features , int(label) ,coord


class TCGADataset(Dataset):
    
    def __init__(self, file_name=None, file_label=None,root=None,persistence=True,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(TCGADataset, self).__init__()

        self.patient_name = file_name
        print('self.file_name:',file_name)
        self.patient_label = file_label
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'h5_files')) if keep_same_psize else os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.slide_label = []
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        a=0
        b=0
        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    self.slide_name.append(torch.load(os.path.join(self.root,'pt_files',_sides[_idx])))
                else:
                    self.slide_name.append(_sides[_idx])
                self.slide_label.append(self.patient_label[i])
        self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]
        for i in range(len(self.slide_label)):
            label = int(self.slide_label[i])
            if label==0:
                a=a+1
            elif label==1:
                b=b+1
            else:
                print('error')
        print('a:',a)
        print('b:',b)


    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        #print('self.slide_name:',file_path)
        label = self.slide_label[idx]

        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path))
        #print('features:',os.path.join(self.root,'pt_files',file_path))
        
        h5_file_path = os.path.join(self.root, 'h5_files',file_path.replace(".pt", ".h5"))  
         
        print('h5_file_path:',h5_file_path)
        with h5py.File(h5_file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][:]
        '''
        coord=0
           
        return features , int(label),coord,file_path.replace(".pt", "")


class C16Dataset(Dataset):
    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset, self).__init__()
        self.file_name = file_name
        print('self.file_name:',file_name)
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]
        a=0
        b=0
        for i in range(len(self.slide_label)):
            label = int(self.slide_label[i])
            if label==0:
                a=a+1
            elif label==1:
                b=b+1
            else:
                print('error')
        print('a:',a)
        print('b:',b)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            #dir_path = os.path.join(self.root,"pt")
            dir_path = os.path.join(self.root,"pt_files")
            file_name=self.file_name[idx]
            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path)
            
            h5_file_path = os.path.join(self.root, 'h5_files',self.file_name[idx]+'.h5')
            with h5py.File(h5_file_path,'r') as hdf5_file:
                coord = hdf5_file['coords'][:]
                #print('coord:',coord)      
              
        label = int(self.slide_label[idx])
        return features , label ,coord,file_name







BENCHMARK_DATA_CACHE = {} 

class Benchmarks(Dataset):
    def __init__(self, file_name, file_label, root, dataset_name='musk1'):
        super(Benchmarks, self).__init__()
        self.file_name = file_name  # 唯一的 Bag IDs
        self.slide_label = [int(_l) for _l in file_label] # 唯一的 Bag Labels
        self.size = len(self.file_name)
        self.root = root 
        self.dataset_name = dataset_name
        
        # 统计标签，这部分是正确的
        a = sum(1 for label in self.slide_label if label == 0)
        b = sum(1 for label in self.slide_label if label == 1)
        print('Total Bags (0/1):', a, b)
        
        # --- 新增：基准数据集的特征预加载 ---
        if self.dataset_name in ['musk1', 'musk2', 'fox', 'tiger', 'elephant']:
            # 找到对应的 CSV 文件路径
            csv_path = os.path.join(root, self.dataset_name + '.csv')
            
            if csv_path not in BENCHMARK_DATA_CACHE:
                print(f"Loading features from {csv_path}...")
                # 假设：header=None, 前两列是 Label 和 ID，其余是特征
                df_full = pd.read_csv(csv_path, header=None)
                df_full.columns = ['Bag_Label', 'Bag_ID'] + [f'Feat_{i}' for i in range(2, df_full.shape[1])]
                
                # 缓存完整的 DataFrame，避免每次都读文件
                BENCHMARK_DATA_CACHE[csv_path] = df_full
            
            self.full_data = BENCHMARK_DATA_CACHE[csv_path]
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 1. 获取当前的 Bag ID 和 Label
        bag_id = self.file_name[idx] # P
        label = self.slide_label[idx] # L
        
        # 2. 从缓存的 DataFrame 中提取该 Bag 的所有实例特征
        # 'Bag_ID' 列与当前的 bag_id 匹配
        bag_df = self.full_data[self.full_data['Bag_ID'] == bag_id]
        
        # 3. 提取特征矩阵
        # 排除前两列 (Bag_Label, Bag_ID)，只保留特征列
        features_df = bag_df.iloc[:, 2:] 
        
        # 转换为 PyTorch Tensor (N x D, N是实例数量, D是特征维度)
        # 使用 .astype(np.float32) 确保数据类型正确
        features = torch.tensor(features_df.values, dtype=torch.float32)

        if torch.isnan(features).any() or torch.isinf(features).any():
            print("🚨 ERROR: Feature contains NaN or Inf!")
            # 抛出异常或进行处理
        # 4. 返回数据
        # 约定返回 (features, label, coord, file_name)
        # coord 在这里没有意义，设为 None 或 0
        return features, label, 0, bag_id
    
import math
def get_lambda_plce(epoch, max_epochs, max_lambda=0.5):
    """Calculates lambda_plce with a cosine annealing schedule, as per paper."""
    return max_lambda * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))

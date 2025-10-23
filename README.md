# MAP-MIL
MAP-MIL: Dual-branch Collaborative Learning of Mask Enhancement and Pseudo-bag Generation for Whole Slide Image Classification

# Train

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

# Test

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






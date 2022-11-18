_base_ = '../base.py'
# dataset 
proj_name = "ImageNet"
concept_root = 'datasets/ImageNet/concepts/'
img_split_path = 'datasets/ImageNet/splits'
img_path = 'datasets/ImageNet/images'
concept_type = "all"

img_ext = ''
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 1000

## data loader
bs = 128
on_gpu = True

# concept select
num_concept = num_cls * 50

# weight matrix fitting
lr = 1e-5
max_epochs = 3000

# weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = False
use_div_loss = False
lambda_l1 = 0.01
lambda_div = 0.005

# CLIP Backbone
clip_model = 'ViT-L/14'
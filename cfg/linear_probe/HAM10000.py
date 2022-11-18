steps = 8
n_runs = 1
img_path = 'datasets/HAM10000/images/'
data_root = 'exp/linear_probe/HAM10000'
img_split_path = 'datasets/HAM10000/splits'
num_cls = 7
unfreeze_clip = False
paper = True

cls_names = ['actinic keratoses', 'basal cell carcinoma', 'benign keratosis-like lesions', 'dermatofibroma', 'melanocytic nevi', 'melanoma', 'vascular lesions']
img_ext=''
clip_model='ViT-L/14'

lr=1e-3
bs=128
n_shots = 8
dataset = 'HAM10000'
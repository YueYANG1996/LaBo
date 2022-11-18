steps = 8
n_runs = 1
img_path = 'datasets/CIFAR10/images/'
data_root = 'exp/linear_probe/CIFAR10'
img_split_path = 'datasets/CIFAR10/splits'
num_cls = 10
unfreeze_clip = False
paper = True
cls_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img_ext=''
clip_model='ViT-L/14'

lr=1e-3
bs=128
n_shots = 8
dataset = 'CIFAR10'
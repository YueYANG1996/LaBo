steps = 8
n_runs = 1
img_path = 'datasets/DTD/images/'
data_root = 'exp/linear_probe/DTD'
img_split_path = 'datasets/DTD/splits'
num_cls = 47
unfreeze_clip = False
paper = True
cls_names = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']
img_ext=''
clip_model='ViT-L/14'

lr=1e-3
bs=128
n_shots = 8
dataset = 'DTD'
steps = 8
n_runs = 1
img_path = 'datasets/RESISC45/images/'
data_root = 'exp/linear_probe/RESISC45'
img_split_path = 'datasets/RESISC45/splits'
num_cls = 45
unfreeze_clip = False
paper = True

cls_names = ['railway station', 'airplane', 'airport', 'baseball diamond', 'basketball court', 'beach', 'bridge', 'chaparral', 'church', 'circular farmland', 'cloud', 'commercial area', 'dense residential', 'desert', 'forest', 'freeway', 'golf course', 'ground track field', 'harbor', 'industrial area', 'intersection', 'island', 'lake', 'meadow', 'medium residential', 'river', 'mobile home park', 'mountain', 'overpass', 'palace', 'parking lot', 'railway', 'rectangular farmland', 'roundabout', 'ship', 'runway', 'sea ice', 'snowberg', 'sparse residential', 'stadium', 'storage tank', 'tennis court', 'terrace', 'thermal power station', 'wetland']

img_ext=''
clip_model='ViT-L/14'

lr=1e-3
bs=128
n_shots = 8
dataset = 'RESISC45'
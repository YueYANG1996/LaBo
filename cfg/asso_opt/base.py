# concept selection
use_mi = True
group_select = True
clip_model = 'ViT-L/14'

# weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = False
use_div_loss = False
lambda_l1 = 0.01
lambda_div = 0.005

use_img_norm = False
use_txt_norm = False

cls_name_init = 'none'
cls_sim_prior = 'none'

remove_cls_name = False
concept_select_fn = None

submodular_weights = 'none'
_base_ = 'food_base.py'
n_shots = 1
data_root = 'exp/asso_opt/food/food_1shot_fac'
lr = 1e-5
bs = 16

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.5]
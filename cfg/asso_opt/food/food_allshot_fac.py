_base_ = 'food_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/food/food_allshot_fac'
lr = 1e-5
bs = 1024

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 5]
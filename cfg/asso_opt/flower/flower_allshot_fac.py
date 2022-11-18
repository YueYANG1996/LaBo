_base_ = 'flower_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/flower/flower_allshot_fac'
lr = 1e-5
bs = 256

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 1]
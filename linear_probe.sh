# $1: dataset, flower or food
# $2: number of shots
# $3: clip model name

python main.py --cfg cfg/linear_probe/$1.py --work-dir exp/linear_probe/$1 --func linear_probe_sklearn_main --DEBUG --cfg-options n_shots=$2 clip_model=$3 bs=256
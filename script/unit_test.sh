cd ../

for config in catboost catboost_gnn gradientboost histgradientboost minimol; do
    CUDA_VISIBLE_DEVICES=$1 python ./main.py \
        --config-name $config \
        job.tasks=['caco2_wang','hia_hou'] \
        job.max_seed=1 \
        +wandb.tags=['debug']
done
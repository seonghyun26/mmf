cd ../

for config in maplight maplight_gnn gradientboost histgradientboost minimol; do
    CUDA_VISIBLE_DEVICES=$1 python ./main.py \
        --config-name $config \
        job.tasks=['caco2_wang','hia_hou'] \
        +wandb.tags=['debug']
done
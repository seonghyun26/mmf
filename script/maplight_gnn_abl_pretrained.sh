cd ../

for pretrained_model in gin_supervised_contextpred gin_supervised_infomax gin_supervised_edgepred jtvae_zinc_no_kl; do
    CUDA_VISIBLE_DEVICES=$1 python ./main.py \
        --config-name maplight_gnn \
        ++model.fingerprint.pretrained.kind=$pretrained_model
done
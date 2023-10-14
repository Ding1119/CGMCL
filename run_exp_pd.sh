DIR=logs/pd_logs
model_select='densenet'
losses_select='Contrastive_loss'
dataset_choice='pd'
image_type='dermatology_images'
for i in DaG
do
    python3 run_gat_gat_skin.py --img_data_dir ${dataset_choice} \
    --skin_type ${image_type} \
    --losses_choice ${losses_select} \
    --model_select ${model_select} \
    --dataset_choice ${dataset_choice} \
    --category ${i} \
    --n_epoch 300 \
     --n_classes 3 >> ${DIR}/abide_${model_select}_${losses_select}_GCN_logs_v3.txt
done
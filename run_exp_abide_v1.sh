DIR=logs
model_select='densenet'
losses_select='Contrastive_loss'
dataset_choice='abide'
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
    --n_classes 2 >> ${DIR}/abide_${model_select}_${losses_select}_Testing_Model_logs.txt
done
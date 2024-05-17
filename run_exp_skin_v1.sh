DIR=logs/skin_logs
model_select='resnet_18'
losses_select='Contrastive_loss'
dataset_choice='skin'
image_type='dermatology_images'
beta=0.4
for i in DaG
do
    python3 run_gat_gat_skin.py --img_data_dir ${dataset_choice} \
    --skin_type ${image_type} \
    --losses_choice ${losses_select} \
    --model_select ${model_select} \
    --dataset_choice ${dataset_choice} \
    --category ${i} \
    --n_epoch 300 \
    --n_classes 3 \
    --beta ${beta}
done


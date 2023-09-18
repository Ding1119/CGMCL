DIR=logs
model_select='densenet'
losses_select='InfoNCE_loss'
dataset_choice='abide'
image_type='dermatology_images'
for i in DaG PIG PN STR VS
do
    python3 run_gat_gat_skin.py --img_data_dir ${dataset_choice} \
    --skin_type ${image_type} \
    --losses_choice ${losses_select} \
    --model_select ${model_select} \
    --dataset_choice ${dataset_choice} \
    --category ${i} \
    --n_epoch 300 \
    --n_classes 3 >> ${DIR}/${dataset_choice}_${image_type}_${model_select}_${losses_select}_Testing_Model_logs.txt
done

DIR=logs/skin_logs
model_select='densenet'
losses_select='Contrastive_loss'
dataset_choice='skin'
image_type='dermatology_images'
for i in DaG PIG PN STR VS
do
    python3 main_v2.py --img_data_dir ${dataset_choice} \
    --skin_type ${image_type} \
    --losses_choice ${losses_select} \
    --model_select ${model_select} \
    --dataset_choice ${dataset_choice} \
    --category ${i} \
    --n_epoch 3 \
    --n_classes 3 #>> ${DIR}/${dataset_choice}_${image_type}_${model_select}_${losses_select}_pos_neg_test_logs.txt
done

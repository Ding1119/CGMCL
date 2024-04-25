DIR=logs/pd_logs
model_select='resnet_18'
losses_select='Contrastive_loss'
dataset_choice='pd'
image_type='dermatology_images'
n_classes=3
exp_mode='normal_mid' #mid_abnormal, normal_mid, normal_abnormal
for i in DaG
do
    python3 run_gat_gat_skin.py --img_data_dir ${dataset_choice} \
    --skin_type ${image_type} \
    --losses_choice ${losses_select} \
    --model_select ${model_select} \
    --dataset_choice ${dataset_choice} \
    --category ${i} \
    --n_epoch 300 \
     --n_classes 2 \
     --exp_mode ${exp_mode} #>> ${DIR}/pd_n_class${n_classes}_${model_select}_${losses_select}_GCN_logs_v3.txt
done
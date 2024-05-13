DIR=logs/pd_logs
model_select='resnet_18'
losses_select='Contrastive_loss'
dataset_choice='pd'
image_type='dermatology_images'
n_classes=2
exp_mode='normal_mid' #mid_abnormal, normal_mid, normal_abnormal

# Clear the output file before starting
output_file="beta_performance.txt"
echo "beta,auc" > $output_file

for beta in 0.1 0.2 0.4 0.6 0.8
do
    for i in DaG
    do
        # Run the python script and capture the AUC performance
        auc=$(python3 run_gat_gat_skin.py --img_data_dir ${dataset_choice} \
        --skin_type ${image_type} \
        --losses_choice ${losses_select} \
        --model_select ${model_select} \
        --dataset_choice ${dataset_choice} \
        --category ${i} \
        --n_epoch 300 \
        --n_classes ${n_classes} \
        --exp_mode ${exp_mode} \
        --beta ${beta} \
        | grep "AUC:" | awk '{print $2}')
        
        # Save the beta and auc to the output file
        echo "${beta},${auc}" >> $output_file
    done
done



# DIR=logs/pd_logs
# model_select='resnet_18'
# losses_select='Contrastive_loss'
# dataset_choice='pd'
# image_type='dermatology_images'
# n_classes=2
# exp_mode='normal_mid' #mid_abnormal, normal_mid, normal_abnormal

# for beta in 0.1 0.2 0.4 0.6 0.8 1
# do
#     for i in DaG
#     do
#         python3 run_gat_gat_skin.py --img_data_dir ${dataset_choice} \
#         --skin_type ${image_type} \
#         --losses_choice ${losses_select} \
#         --model_select ${model_select} \
#         --dataset_choice ${dataset_choice} \
#         --category ${i} \
#         --n_epoch 100 \
#         --n_classes ${n_classes} \
#         --exp_mode ${exp_mode} \
#         --beta ${beta}
#     done
# done

DIR=logs
model_select='densenet'
losses_select='SAC_loss'
image_type='clinical_images'
for i in DaG
do
    python3 run_gat_gat_skin.py --skin_type ${image_type} \
    --meta_data_dir "abide" \
    --model_select ${model_select} \
    --losses_choice ${losses_select} \
    --classes ${i} \
    --n_epoch 300 \
    --n_classes 2 >> ${DIR}/abide_${model_select}_${losses_select}_Testing_Model_logs.txt

done

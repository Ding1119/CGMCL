DIR=logs/new_exp
model_select='densenet'
losses_select='Contrastive_loss'
image_type='dermatology_images'
for i in DaG PIG PN STR VS
do
    python3 run_gat_gat_skin.py --skin_type ${image_type} \
    --meta_data_dir "/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok" \
    --model_select ${model_select} \
    --losses_choice ${losses_select} \
    --classes ${i} \
    --n_epoch 300 \
    --n_classes 3 >> ${DIR}/skin_${image_type}_${model_select}_${losses_select}_Testing_Model_Adj_Img_Kmean_logs.txt

done

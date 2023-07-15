DIR=logs
losses_select='MGEC_loss'
image_type='clinical_images'
for i in STR VS
do
    python3 run_gat_gat_skin.py --skin_type ${image_type} \
    --meta_data_dir "/Users/test/Documents/Contrastive_PD/skin_dataset_ok/meta_ok/" \
    --losses_choice ${losses_select} \
    --classes ${i} \
    --n_epoch 8 \
    --n_classes 3
    # --n_classes 3 >> logs/skin_${image_type}_${losses_select}_logs.txt
done
#!/bin/bash
# 定義變量
DIR=logs/pd_logs
model_select='resnet_50'
losses_select='Contrastive_loss'
dataset_choice='pd'
image_type='dermatology_images'
n_classes=2
exp_mode='normal_abnormal' #normal_mid, mid_abnormal, normal_abnormal
num_experiments=5

# 確保存儲結果的目錄存在
mkdir -p ${DIR}

# 運行實驗
for i in DaG
do
  # 初始化結果檔名
  output_file="${DIR}/pd_${exp_mode}_n_class${n_classes}_${model_select}_${losses_select}_GCN_logs_v3_${i}.txt"
  > ${output_file}  # 清空檔案內容

  # 進行五次實驗
  for (( j=1; j<=num_experiments; j++ ))
  do
    echo "Running experiment $j for category $i"
    python3 run_gat_gat_skin.py --img_data_dir ${dataset_choice} \
    --skin_type ${image_type} \
    --losses_choice ${losses_select} \
    --model_select ${model_select} \
    --dataset_choice ${dataset_choice} \
    --category ${i} \
    --n_epoch 300 \
    --n_classes 2 \
    --exp_mode ${exp_mode} >> ${output_file}
  done

  # 讀取和計算結果的平均值與標準差
  echo "Results for category $i:" >> ${output_file}
  awk '/AUC|Accuracy|Sensitivity|Specificity|PPV|NPV/{print $0}' ${output_file} | awk '
  BEGIN{
    count=0;
    sum_auc=sum_acc=sum_sen=sum_spe=sum_ppv=sum_npv=0;
    sq_sum_auc=sq_sum_acc=sq_sum_sen=sq_sum_spe=sq_sum_ppv=sq_sum_npv=0;
  }
  /AUC/{auc=$2; sum_auc+=auc; sq_sum_auc+=auc*auc;}
  /Accuracy/{acc=$2; sum_acc+=acc; sq_sum_acc+=acc*acc;}
  /Sensitivity/{sen=$2; sum_sen+=sen; sq_sum_sen+=sen*sen;}
  /Specificity/{spe=$2; sum_spe+=spe; sq_sum_spe+=spe*spe;}
  /PPV/{ppv=$2; sum_ppv+=ppv; sq_sum_ppv+=ppv*ppv;}
  /NPV/{npv=$2; sum_npv+=npv; sq_sum_npv+=npv*npv; count++;}
  END{
    mean_auc=sum_auc/count; sd_auc=sqrt(sq_sum_auc/count - mean_auc*mean_auc);
    mean_acc=sum_acc/count; sd_acc=sqrt(sq_sum_acc/count - mean_acc*mean_acc);
    mean_sen=sum_sen/count; sd_sen=sqrt(sq_sum_sen/count - mean_sen*mean_sen);
    mean_spe=sum_spe/count; sd_spe=sqrt(sq_sum_spe/count - mean_spe*mean_spe);
    mean_ppv=sum_ppv/count; sd_ppv=sqrt(sq_sum_ppv/count - mean_ppv*mean_ppv);
    mean_npv=sum_npv/count; sd_npv=sqrt(sq_sum_npv/count - mean_npv*mean_npv);
    printf("Average and SD of AUC: %f ± %f\n", mean_auc, sd_auc);
    printf("Average and SD of Accuracy: %f ± %f\n", mean_acc, sd_acc);
    printf("Average and SD of Sensitivity: %f ± %f\n", mean_sen, sd_sen);
    printf("Average and SD of Specificity: %f ± %f\n", mean_spe, sd_spe);
    printf("Average and SD of PPV: %f ± %f\n", mean_ppv, sd_ppv);
    printf("Average and SD of NPV: %f ± %f\n", mean_npv, sd_npv);
  }' >> ${output_file}
done
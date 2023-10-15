import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import roc_curve,auc
from scipy import interp
from sklearn.neighbors import NearestNeighbors
from itertools import cycle
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from sklearn.cluster import KMeans
import torchvision.models as models
from torch.autograd import Variable
from sklearn.metrics import classification_report
from sklearn import preprocessing
from utils import build_knn_graph, label_return, plot_auc, print_auc, calculate_metrics_new, run_eval
from model_resnet_skin import Projection, Model_SKIN, CNN
from model_resnet_abide import Projection, Model_ABIDE, CNN
from model_resnet_pd import Projection, Model_PD, CNN
from losses import WeightedCrossEntropyLoss, contrastive_loss, info_loss, MGECLoss, SACLoss
import argparse
from data_handlder.load_dataset import dataloader
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, auc
from sklearn.metrics import plot_roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve, auc, accuracy_score, precision_recall_fscore_support, jaccard_score


from sklearn.model_selection import StratifiedKFold
from skimage import data_dir,io,color

def dataloader_cv(datadir, skin_type, num_folds=5):
    
    if datadir == 'pd':
        path_img = '/home/jding/Documents/PD_contrastive_research_0817/spect_513_data' + '/'
        path_meta = '/home/jding/Documents/PD_contrastive_research_0817/spect_513_data' + '/'
        coll = io.ImageCollection('/home/jding/Documents/PD_contrastive_research_0817/spect_513_data/spect_img_a2/*.jpg')

        raw_image = io.concatenate_images(coll)

        raw_meta = pd.read_csv(path_meta + 'label_513.csv')
        label_630_id = raw_meta[raw_meta['ID'] < 634]
        raw_data = raw_image[label_630_id['Python_ID']] / 255
        raw_data = torch.tensor(raw_data).transpose(1, 3)
        raw_data = np.array(raw_data)
        raw_patients_feature_412 = np.asarray(label_630_id.iloc[:, 8:20])
#         import pdb;pdb.set_trace()
        # 创建 StratifiedKFold 对象
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        image_data_train_list = []
        feature_data_train_list = []
        adj_train_img_list = []
        adj_f_knn_train_list = []
        image_data_test_list = []
        test_feature_data_list = []
        adj_test_img_list = []
        adj_f_knn_test_list = []
        y_train_list = []
        y_test_list = []
#         import pdb;pdb.set_trace()
        for train_index, test_index in skf.split(raw_data, label_630_id['Label_2'].values):
            y_train = label_630_id['Label_2'].values[train_index]
            y_test = label_630_id['Label_2'].values[test_index]
            # 根据每个折叠的索引提取相应的训练和测试数据
            raw_image_train = raw_data[train_index]
            raw_image_test = raw_data[test_index]

            raw_f_train = raw_patients_feature_412[train_index]
            raw_f_test = raw_patients_feature_412[test_index]

            image_data_train = torch.from_numpy(raw_image_train).float()
            feature_data_train = torch.from_numpy(raw_f_train).float()
            image_data_flatten = torch.flatten(image_data_train, start_dim=1)
            adj_train_img = build_knn_graph(image_data_flatten, len(train_index)).float()

            image_data_test = torch.from_numpy(raw_image_test).float()
            data_features_test = raw_f_test
            test_feature_data = torch.from_numpy(data_features_test).float()

            image_data_test_flatten = torch.flatten(image_data_test, start_dim=1)
            adj_test_img = build_knn_graph(image_data_test_flatten, len(test_index)).float()
            adj_f_knn_train = build_knn_graph(raw_f_train, len(train_index)).float()
            adj_f_knn_test = build_knn_graph(raw_f_test, len(test_index)).float()

            # 将每个折叠的数据添加到列表中
            y_train_list.append(y_train)
            y_test_list.append(y_test)
            image_data_train_list.append(image_data_train)
            feature_data_train_list.append(feature_data_train)
            adj_train_img_list.append(adj_train_img)
            adj_f_knn_train_list.append(adj_f_knn_train)
            image_data_test_list.append(image_data_test)
            test_feature_data_list.append(test_feature_data)
            adj_test_img_list.append(adj_test_img)
            adj_f_knn_test_list.append(adj_f_knn_test)

    return image_data_train_list, feature_data_train_list, adj_train_img_list, \
        adj_f_knn_train_list, image_data_test_list, test_feature_data_list, \
        adj_test_img_list, adj_f_knn_test_list, y_train_list ,y_test_list
    


import json


def train_eval(datadir,skin_type, loss_select, model_select , dataset_choice ,category, epoch, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    torch.cuda.is_available()
    
    class_name = 'Dag'
    if model_select == 'resnet_18':
        model_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # resnet = models.resnet18(pretrained=True)
    elif model_select == 'resnet_34':
         model_net = models.resnet34(pretrained=True)
    elif model_select == 'resnet_50':
        model_net = models.resnet50(pretrained=True)
    elif model_select == 'densenet':
         model_net = models.densenet121(pretrained=True)
        
    #resnet = models.alexnet(pretrained=True)
    
    model_net = model_net.to(device)
    # 将最后一层的输出维度修改为类别数目
    num_classes = 1024
    
    # num_features = model_net.fc.in_features #512 # Resnet
    num_features = model_net.classifier.in_features # Desnet101
    # import pdb;pdb.set_trace()
    # model_net.fc = nn.Linear(num_features, num_classes) #512 # Resnet
    # model_net.fc = model_net.fc.to(device) #512 # Resnet
    model_net.classifier = nn.Linear(num_features, num_classes) #desnet
    
    image_data_train_list, feature_data_train_list, adj_train_img_list, \
    adj_f_knn_train_list, image_data_test_list, test_feature_data_list, \
    adj_test_img_list, adj_f_knn_test_list, y_train_list ,y_test_list = dataloader_cv(datadir,skin_type, num_folds=5)
 
#     image_data_train, feature_data_train, adj_train_img, adj_f_knn_train, image_data_test, test_feature_data, adj_test_img, adj_f_knn_test = dataloader(datadir,skin_type)
    
    
    all_aucs = []
    all_fpr = []
    all_tpr = []
    all_roc_auc = []
    false_positives = []  # 用于存储 False Positives 的样本 ID
    false_negatives = []  # 用于存储 False Negatives 的样本 ID

    for fold in range(len(image_data_train_list)):
        print(f"Fold {fold + 1}/{len(image_data_train_list)}:")
        
        image_data_train = image_data_train_list[fold]
        feature_data_train = feature_data_train_list[fold]
        adj_train_img = adj_train_img_list[fold]
        adj_f_knn_train = adj_f_knn_train_list[fold]
        image_data_test = image_data_test_list[fold]
        test_feature_data = test_feature_data_list[fold]
        adj_test_img = adj_test_img_list[fold]
        adj_f_knn_test = adj_f_knn_test_list[fold]
        
        y_train = y_train_list[fold]
        y_test = y_test_list[fold]
     
        image_data_train = image_data_train.to(torch.float32).to(device)
        feature_data_train = feature_data_train.to(torch.float32).to(device)
        # import pdb;pdb.set_trace()
        adj_train_img = adj_train_img.to(torch.float32).to(device)
        adj_f_knn_train = adj_f_knn_train.to(torch.float32).to(device)
        image_data_test = image_data_test.to(torch.float32).to(device)
        test_feature_data = test_feature_data.to(torch.float32).to(device)
        adj_test_img = adj_test_img.to(torch.float32).to(device)
        adj_f_knn_test = adj_f_knn_test.to(torch.float32).to(device)

        # y_train = y_train.to(device)
        # y_test = y_test.to(device)
        
        # import pdb;pdb.set_trace()


        projection = Projection(262, 3)
        if datadir == 'skin':
            model = Model_SKIN(projection, model_net, n_classes).to(device)
        elif datadir == 'abide':
            model = Model_ABIDE(projection, model_net, n_classes).to(device)
        elif datadir == 'pd':
            model = Model_PD(projection, model_net, n_classes).to(device, dtype=torch.float32)

        class_weights = torch.full((1,n_classes),0.5).view(-1)
        criterion1 = WeightedCrossEntropyLoss(weight=class_weights)

        if loss_select == 'Contrastive_loss':
             criterion2 = contrastive_loss

        elif loss_select == 'MGEC_loss':
            criterion2 = MGECLoss()

        elif loss_select == 'InfoNCE_loss':
            criterion2 = info_loss

        elif loss_select == 'SAC_loss':
            criterion2 = SACLoss()

    # criterion3 = loss_dependence
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        n_epochs = epoch

        training_range = tqdm(range(n_epochs))

        for epoch in training_range:
            optimizer.zero_grad()
            # cnn_z  =  cnn_encoder(image_data)
            # 前向传播

            # image_data_train = image_data_train.to(device)
            # feature_data_train = feature_data_train.to(device)
            # adj_train_img = adj_train_img.to(device)
            # adj_f_knn_train = adj_f_knn_train.to(device)
            
            output1, output2, emb = model(image_data_train , feature_data_train,adj_train_img, adj_f_knn_train)
      
            
            y = torch.from_numpy(y_train).to(torch.int64).to(device)

            loss_ce1 = criterion1(output1, y)
            loss_ce2 = criterion1(output2, y)
            alpha = 0.4

            if loss_select == 'Contrastive_loss':
                adj = adj_train_img +  adj_f_knn_train
                diag = torch.diag(adj.sum(dim=1))
                loss_extra = criterion2( emb, adj_train_img, adj_f_knn_train, y, output1, output2, diag).to(device)
                loss = (1-alpha)*(loss_ce1 + loss_ce2) + alpha* loss_extra

            elif loss_select == 'MGEC_loss':
                adj = adj_train_img +  adj_f_knn_train
                diag = torch.diag(adj.sum(dim=1))
                loss_extra = criterion2(output1, output2, adj, diag )
                loss = (1-alpha)*(loss_ce1+loss_ce2) + alpha* loss_extra
                #loss = loss_extra

            elif loss_select == 'InfoNCE_loss':
                loss_extra = criterion2( emb, adj_train_img, adj_f_knn_train, y)
                loss = (1-alpha)*(loss_ce1+loss_ce2) + alpha* loss_extra

            elif loss_select == 'SAC_loss':    
                adj = adj_train_img +  adj_f_knn_train
                diag = torch.diag(adj.sum(dim=1))
                loss_extra = criterion2(emb, adj)
                loss = (1-alpha)*(loss_ce1+loss_ce2) + alpha* loss_extra
            elif loss_select == 'only_CE':
                loss = loss_ce1 + loss_ce2


            loss.backward()
            optimizer.step()
            # torch.save(model,f'{skin_type}_{epoch}epoch_save.pt')
            # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))

        model.eval()
        with torch.no_grad():
            # import pdb;pdb.set_trace()
#             image_data_test = image_data_test.to(device)
#             test_feature_data = test_feature_data.to(device)
#             adj_test_img = adj_test_img.to(device)
#             adj_f_knn_test = adj_f_knn_test.to(device)
            # import pdb;pdb.set_trace()
            test_output1, test_output2, emb  = model(image_data_test, test_feature_data , adj_test_img, adj_f_knn_test )

            # test_output1  = model(test_image_data, test_adjacency_matrix, adj_test_img )
            m = nn.Softmax(dim=1)
            # import pdb;pdb.set_trace()
            test_output = test_output1 + test_output2
        #     test_output = emb

            #z = test_output1 + test_output2

            #num_clusters = 3
            #kmeans = KMeans(n_clusters=num_clusters)

            #cluster_labels = kmeans.fit_predict(z.cpu().data.numpy())
            #cluster_labels = torch.tensor(cluster_labels,dtype=torch.int64).to(device)



            pred =  m(test_output).argmax(dim=1)
            pred_prob = m(test_output)[:,1]
            #pred = cluster_labels
            # test_output = test_output.argmax(dim=1)


            # y_test = torch.empty(100).random_(2)
        #     y_test = torch.tensor(label_3_test).to(device)
#             y_test = torch.from_numpy(label_return(dataset_choice ,class_name, "test")).to(device)

            # import pdb;pdb.set_trace()
            y_test = torch.from_numpy(y_test).to(torch.int64).to(device)
            correct = (pred  == y_test).sum().item()
            accuracy = correct / len(y_test)
#             import pdb;pdb.set_trace()
            fpr, tpr, thresholds = roc_curve(y_test.cpu().detach().numpy(), pred_prob.cpu().detach().numpy())
            roc_auc = auc(fpr, tpr)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_roc_auc.append(roc_auc)
            all_aucs.append(roc_auc)
            
            false_positive_indices = np.where((y_test == 0) & (pred.cpu().detach().numpy() == 1))[0]
            false_negative_indices = np.where((y_test == 1) & (pred.cpu().detach().numpy() == 0))[0]

            false_positives.append(false_positive_indices.tolist())
            false_negatives.append(false_negative_indices.tolist())
            
            cm = confusion_matrix(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy())
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Fold {fold + 1}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')



#             import pdb;pdb.set_trace()
            print(f"++++Use {model_select} model+++")
            print(calculate_metrics_new(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
            print("Loss:", loss_select, "class_name",class_name,"Accuracy:", accuracy)
            print(classification_report(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
            if datadir == 'pd':
                accuracy, precision, recall, fscore, sensivity, specificity, nmi, ari = run_eval(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy())
                print("acc:",accuracy, "precision:", precision,"recall:", recall,"fscore:", fscore,"sensitivity:", sensivity,"specificity:", specificity, "nmi", nmi, "ari", ari)
                # plot_ROC(pred.cpu().detach().numpy() , y_test.cpu().detach().numpy(), 3, classes, skin_type, loss_select)
                # print_auc(pred.cpu().detach().numpy() , y_test.cpu().detach().numpy(), 3, category, skin_type, loss_select)


    plt.figure()
    for i in range(5):
        plt.plot(all_fpr[i], all_tpr[i], lw=2, label=f'ROC curve (area = {all_roc_auc[i]:.2f}) - Fold {i + 1}')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - All Folds')
    plt.legend(loc="lower right")
    plt.show()   
    
    # 计算 AUC 的平均值和标准差
    avg_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)

    print(f'Average AUC: {avg_auc:.2f}')
    print(f'Standard Deviation of AUC: {std_auc:.2f}')

    # 存储 False Positives 和 False Negatives 到 JSON 文件
#     result_data = {
#         'FalsePositives': false_positives,
#         'FalseNegatives': false_negatives
#     }

#     with open('false_samples.json', 'w') as json_file:
#         json.dump(result_data, json_file, indent=4)

            
def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img_data_dir', type=str, default='pd')
#     parser.add_argument('--skin_type', type=str, default='dermatology_images')
#     #parser.add_argument('--meta_data_dir', type=str, default='/home/feng/jeding/PD_contrastive_research_0817/meta_ok/')
#     parser.add_argument('--model_select', type=str, default='densenet')
#     parser.add_argument('--losses_choice', type=str, default='Contrastive_loss')
#     parser.add_argument('--dataset_choice', type=str, default='pd')
#     parser.add_argument('--category', type=str)
#     parser.add_argument('--n_epoch', type=int, default=300)
#     parser.add_argument('--n_classes', type=int, default=2)
    
#     args = parser.parse_args()

    img_data_dir = 'pd'
    skin_type = 'dermatology_images'
    losses_choice = 'Contrastive_loss'
    model_select = 'densenet'
    dataset_choice = 'pd'
    category = 'your_category'  # 设置正确的类别值
    n_epoch = 300
    n_classes = 2

    train_eval(img_data_dir, skin_type, losses_choice, model_select, dataset_choice, category, n_epoch, n_classes)
#     train_eval(args.img_data_dir, args.skin_type, args.losses_choice, args.model_select, args.dataset_choice,args.category, args.n_epoch, args.n_classes)


if __name__ == '__main__':
    main()

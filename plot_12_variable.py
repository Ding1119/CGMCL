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
        path_img = '/home/feng/jeding/PD_contrastive_research_0817/spect_513_data' + '/'
        path_meta = '/home/feng/jeding/PD_contrastive_research_0817/spect_513_data' + '/'
        coll = io.ImageCollection('/home/feng/jeding/PD_contrastive_research_0817/spect_513_data/spect_img_a2/*.jpg')
        
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
        for train_index, test_index in skf.split(raw_data, label_630_id['Lebel_3'].values):
            y_train = label_630_id['Lebel_3'].values[train_index]
            y_test = label_630_id['Lebel_3'].values[test_index]
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from sklearn.cluster import KMeans
from keras.applications.mobilenet import MobileNet, preprocess_input
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
import seaborn as sns
#os.environ['CUDA_LAUNCH_BLOCKING'] = "0"


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 1024)
        
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    
class GraphConvolution_img(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution_img, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        # self.linear2 = nn.Linear( 65536, 12)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x, adjacency):
        # x = x.T @ self.linear(x)
       
        x = self.linear2(x)
        # import pdb;pdb.set_trace()
        x = torch.matmul(adjacency, x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)

        return x
    
class GraphConvolution_f(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution_f, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        # self.linear2 = nn.Linear( 65536, 12)
        self.linear2 = nn.Linear(256, 12)
        self.linear3 = nn.Linear(12, 3)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x, adjacency):
        # x = x.T @ self.linear(x)
        
        # x = self.linear2(x)
        # import pdb;pdb.set_trace()
        x = torch.matmul(adjacency, x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        # import pdb;pdb.set_trace()
        return x
    
class AttentionNetwork_img(nn.Module):
    def __init__(self, num_features):
        super(AttentionNetwork_img, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        # x = F.dropout(x, p=0.5)
        x = self.sigmoid(x)
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        attention_weights = self.sigmoid(x)
        return attention_weights
    
class AttentionNetwork_f(nn.Module):
    def __init__(self, num_features):
        super(AttentionNetwork_f, self).__init__()
        self.fc1 = nn.Linear(num_features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        # x = F.dropout(x, p=0.5)
        x = self.sigmoid(x)
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        attention_weights = self.sigmoid(x)
        return attention_weights


class Projection(nn.Module):

    def __init__(self, input_dim, hid_dim):
        super(Projection, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = nn.ReLU()
        self.layernorm = nn.LayerNorm(hid_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x
    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

    def forward(self, X, adj):
        h = self.W(X)
        N = h.size(0)
        
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * h.size(1))
        e = F.relu(self.a(a_input).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        
        h_prime = torch.matmul(attention, h)
        return h_prime

# 定義 Graph Attention Network
class GAT_img(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GAT_img, self).__init__()
        self.attention1 = GraphAttentionLayer(in_features, hidden_features)
        self.attention2 = GraphAttentionLayer(hidden_features, num_classes)

    def forward(self, X, adj):
        h = self.attention1(X, adj)
        h = F.relu(h)
        logits = self.attention2(h, adj)
        return logits
    
class GAT_f(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GAT_f, self).__init__()
        self.attention1 = GraphAttentionLayer(in_features, hidden_features)
        self.attention2 = GraphAttentionLayer(hidden_features, num_classes)

    def forward(self, X, adj):
        # h =  torch.matmul(adj, X)
        
        h = self.attention1(X, adj)
        h = F.relu(h)
        logits = self.attention2(h, adj)
        return logits
    
class Model_PD_Weight(nn.Module):
    def __init__(self, projection, input_resnet, n_classes):
        super(Model_PD_Weight, self).__init__()
        #self.encoder = CNNEncoder()
        self.cnn_encoder = CNN()
        self.input_resnet = input_resnet
        self.n_classes = n_classes
        self.kmeans = KMeans(n_clusters=3) 
        # self.cnn_encoder = DeepEncoder() #torch.Size([300, 3, 64, 64])
        # self.cnn_encoder = CNNEncoder()
        # self.vgg_encoder = nn.Sequential(*list(vgg16.features.children()))
        self.gcn_img = GraphConvolution_img(64, 3)
        self.gcn_f = GraphConvolution_f(12, 3)
        self.gat_img = GAT_img(1024,256,3)
        self.gat_f = GAT_f(12,5,3)
        self.projection = projection
   
        # self.linear_vgg1 = nn.Linear(65536, 1024)
        # self.linear_vgg2 = nn.Linear(1024, 256)
        self.linear1 = nn.Linear(259, 2)
        self.linear2 = nn.Linear(256, 2)
        self.linear3 = nn.Linear(1030, 256)
        self.linear4 = nn.Linear(256,self.n_classes)
        self.linear_f = nn.Linear(18,self.n_classes)
        self.attention_f = AttentionNetwork_f(15)
        self.attention_img = AttentionNetwork_img(1027)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_f,adjacency_img, adjacency_f, epoch):
        # import pdb;pdb.set_trace()
        # x = self.encoder(x) #torch.Size([300, 65536])
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        x_encoder = self.input_resnet(x)
        # x_encoder = self.cnn_encoder(x)
        
        #x_encoder = self.cnn_encoder(x)

        x_gat_img = self.gat_img(x_encoder, adjacency_img)
        # x_encoder = self.vgg_encoder(x)
        # x_encoder = torch.flatten(x_encoder, start_dim=1)
        # x_encoder = self.linear_vgg1(x_encoder)
        # x_encoder = self.linear_vgg2(x_encoder)
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        
        x_gat_f = self.gat_f(x_f, adjacency_img)
        # import pdb;pdb.set_trace()
        x_gcn_f = self.gcn_f(x_f, adjacency_f)
        # import pdb;pdb.set_trace()
        # kmeans_output_f = self.kmeans.fit_transform(x_gcn_f.detach().numpy())
        # kmeans_tensor_f = torch.from_numpy(kmeans_output_f).float().to(x_gcn_f.device)
        
        x_gcn_img = self.gcn_img(x_encoder, adjacency_img)
        # kmeans_output_img = self.kmeans.fit_transform(x_gcn_img.detach().numpy())
        # kmeans_tensor_img = torch.from_numpy(kmeans_output_img).float().to(x_gcn_img.device)

        x_f_fusion = torch.cat((x_f, x_gat_f), 1)
        x_img = torch.cat((x_encoder, x_gat_img), 1)
       
        
        w_att_f = self.attention_f(x_f_fusion)
        w_att_img = self.attention_img(x_img)
        
        attended_gat_f = x_gat_f * w_att_f
        attended_gat_img = x_gat_img * w_att_img
#         import pdb;pdb.set_trace()
#         if epoch == 50:
#             attended_gat_f = x_gat_f * w_att_f
#             attended_gat_img = x_gat_img * w_att_img

#             np.save(f'epoch_{epoch}_f_weight', attended_gat_f.cpu().detach().numpy())
#             np.save(f'epoch_{epoch}_img_weight', attended_gat_img.cpu().detach().numpy())
#         elif epoch ==100:
#             attended_gat_f = x_gat_f * w_att_f
#             attended_gat_img = x_gat_img * w_att_img

#             np.save(f'epoch_{epoch}_f_weight', attended_gat_f.cpu().detach().numpy())
#             np.save(f'epoch_{epoch}_img_weight', attended_gat_img.cpu().detach().numpy())
#         elif epoch ==299:
#             attended_gat_f = x_gat_f * w_att_f
#             attended_gat_img = x_gat_img * w_att_img

#             np.save(f'epoch_{epoch}_f_weight', attended_gat_f.cpu().detach().numpy())
#             np.save(f'epoch_{epoch}_img_weight', attended_gat_img.cpu().detach().numpy())
#         else:


        attended_gat_f = x_gat_f * w_att_f
        attended_gat_img = x_gat_img * w_att_img
        
        
        
        x_f_att = torch.cat((x_f_fusion, attended_gat_f), 1) # torch.Size([300, 17])
        x_img_att = torch.cat((x_img, attended_gat_img), 1) # torch.Size([300, 1029])
        # 這部份Journal可以寫數學式子w
 
        emb1  = self.linear_f(x_f_att)
        # emb1  = self.linear4(emb1)
        
        emb2  = self.linear3(x_img_att)
        emb2  = self.linear4(emb2)
        #import pdb;pdb.set_trace()
        # x1 = self.linear1(x_f)
      
        
        # # x2 = self.linear2(x_encoder)
        # x2 = self.linear1(x_img)

        # x1 = self.softmax(x1)
        # x2 = self.softmax(x2)

        # x1 = self.linear1(attended_gcn)
        z = emb1 + emb2
        
        
  
        z  = self.softmax(z).to(torch.float32)
        
        #import pdb;pdb.set_trace()
        
       
        return emb1, emb2, z
    
    def nonlinear_transformation(self, h):
        z = self.projection(h)
        return z
    

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
    
    num_features = model_net.fc.in_features #512 # Resnet
#     num_features = model_net.classifier.in_features # Desnet101
    # import pdb;pdb.set_trace()
    model_net.fc = nn.Linear(num_features, num_classes) #512 # Resnet
    model_net.fc = model_net.fc.to(device) #512 # Resnet
#     model_net.classifier = nn.Linear(num_features, num_classes) #desnet
    
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

        n_epochs = epoch
        projection = Projection(262, 3)
        if datadir == 'skin':
            model = Model_SKIN(projection, model_net, n_classes).to(device)
        elif datadir == 'abide':
            model = Model_ABIDE(projection, model_net, n_classes).to(device)
        elif datadir == 'pd':
            model = Model_PD_Weight(projection, model_net, n_classes).to(device, dtype=torch.float32)

        class_weights = torch.full((1,n_classes),0.5).view(-1)
        criterion1 = WeightedCrossEntropyLoss(weight=class_weights)

        if loss_select == 'Contrastive_loss':
#              criterion2 = contrastive_loss
            criterion2 = contrastive_loss

        elif loss_select == 'MGEC_loss':
            criterion2 = MGECLoss()

        elif loss_select == 'InfoNCE_loss':
            criterion2 = info_loss

        elif loss_select == 'SAC_loss':
            criterion2 = SACLoss()

    # criterion3 = loss_dependence
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        

        training_range = tqdm(range(n_epochs))

        for n_epoch in training_range:
            optimizer.zero_grad()
            # cnn_z  =  cnn_encoder(image_data)
            # 前向传播

            # image_data_train = image_data_train.to(device)
            # feature_data_train = feature_data_train.to(device)
            # adj_train_img = adj_train_img.to(device)
            # adj_f_knn_train = adj_f_knn_train.to(device)
            
            output1, output2, emb = model(image_data_train , feature_data_train,adj_train_img, adj_f_knn_train, n_epoch)
      
            
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
            test_output1, test_output2, emb  = model(image_data_test, test_feature_data , adj_test_img, adj_f_knn_test, n_epoch )

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
#             fpr, tpr, thresholds = roc_curve(y_test.cpu().detach().numpy(), pred_prob.cpu().detach().numpy())
#             roc_auc = auc(fpr, tpr)
#             all_fpr.append(fpr)
#             all_tpr.append(tpr)
#             all_roc_auc.append(roc_auc)
#             all_aucs.append(roc_auc)
#             import pdb;pdb.set_trace()
            
            false_positive_indices = np.where((y_test.cpu().numpy() == 0) & (pred.cpu().detach().numpy() == 1))[0]
            false_negative_indices = np.where((y_test.cpu().numpy()  == 1) & (pred.cpu().detach().numpy() == 0))[0]

            false_positives.append(false_positive_indices.tolist())
            false_negatives.append(false_negative_indices.tolist())
            
            cm = confusion_matrix(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy())
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Fold {fold + 1}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            
            plt.figure()

#             plt.figure(figsize=(8, 6))

            # 分別取出不同類別的資料
#             normal_data = test_output[pred == 0].cpu().detach().numpy()
#             neurodegenerative_data = test_output[pred == 1].cpu().detach().numpy()

       

#             normal_data = test_feature_data[pred == 0].cpu().detach().numpy()
#             neurodegenerative_data = test_feature_data[pred == 1].cpu().detach().numpy()
            normal_data = test_feature_data[y_test == 0].cpu().detach().numpy()
            neurodegenerative_data = test_feature_data[y_test == 1].cpu().detach().numpy()
            other_data = test_feature_data[y_test == 2].cpu().detach().numpy()

            # Labels and titles setup
#             x_labels = ["S-R", "AP-R", "PP-R", "C-R", "P/C-R", "PA"]
#             y_labels = ["S-L", "AP-L", "PP-L", "C-L", "P/C-L", "CA"]

            x_labels = ["Right stratal SBR", "Right anterior putaminal SBR", "Right posterior putaminal SBR",
                        "Right caudate SBR", "Right putamen/caudate ratio", "Putamen asymmetry"]
            y_labels = ["Left stratal SBR", "Left anterior putaminal SBR", "Left posterior putaminal SBR",
                        "Left caudate SBR", "Left putamen/caudate ratio", "Caudate asymmetry"]
            titles = ["Striatal SBR", "Anterior putaminal SBR", "Posterior putaminal SBR",
                      "Caudate SBR", "Putamen/caudate ratio", "Putamen and caudate asymmetry"]

            # Visualization
            n_rows = 2
            n_cols = 3

            # Ensure there are enough columns in the data for plotting
            if test_feature_data.shape[1] >= 2:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
                axes = axes.flatten()
                plt.tight_layout(pad=4.0)

                for i in range(len(x_labels)):
                    ax = axes[i]
                    # Plot each category
                    ax.scatter(normal_data[:, i], normal_data[:, i + 1], c='green', marker='o', edgecolors='black', label='Normal')
                    ax.scatter(neurodegenerative_data[:, i], neurodegenerative_data[:, i + 1], c='red', marker='^', edgecolors='black', label='Abnormal')
                    if other_data.size > 0:
                        ax.scatter(other_data[:, i], other_data[:, i + 1], c='#FFA07A', marker='s', edgecolors='black', label='MA')
#                         ax.scatter(other_data[:, i], other_data[:, i + 1], c='blue', marker='s', edgecolors='black', label='Other')
                         
                    ax.set_xlabel(x_labels[i % len(x_labels)], fontsize=18)
                    ax.set_ylabel(y_labels[i % len(y_labels)], fontsize=18)
                    ax.set_title(titles[i % len(titles)], fontsize=20)
                    ax.legend(fontsize=10)
                plt.subplots_adjust(hspace=0.4, wspace=0.4)
#                 plt.savefig("plot_output.png", format='png', dpi=300)  # Increased DPI for better resolution
                plt.savefig("plot_output.pdf", format='pdf')
#                 plt.close(fig)  # Close the figure to free up memory

                
#                 plt.show()
                
                
            else:
                print("Insufficient columns in data to plot scatter plots.")


#     plt.figure()
#     for i in range(5):
#         plt.plot(all_fpr[i], all_tpr[i], lw=2, label=f'ROC curve (area = {all_roc_auc[i]:.2f}) - Fold {i + 1}')

#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC - All Folds')
#     plt.legend(loc="lower right")
#     plt.show()   
    
#     # 计算 AUC 的平均值和标准差
#     avg_auc = np.mean(all_aucs)
#     std_auc = np.std(all_aucs)

#     print(f'Average AUC: {avg_auc:.2f}')
#     print(f'Standard Deviation of AUC: {std_auc:.2f}')
    
#     result_data = {
#         'FalsePositives': false_positives,
#         'FalseNegatives': false_negatives
#     }

        
    # 存储 False Positives 和 False Negatives 到 JSON 文件

#     with open('false_samples.json', 'w') as json_file:
#         json.dump(result_data, json_file, indent=4)
    
#     datadir = '/home/jding/Documents/PD_contrastive_research_0817/spect_513_data/spect_img_a2'
    
#     target_false_positives_dir = '/home/jding/Documents/PD_contrastive_research_0817/saved_fig/target_false_positives'  # 指定目标文件夹
#     os.makedirs(target_false_positives_dir, exist_ok=True)
#     copy_images(datadir, target_false_positives_dir, false_positives)

#     # 复制 False Negatives 的影像到目标文件夹
#     target_false_negatives_dir = '/home/jding/Documents/PD_contrastive_research_0817/saved_fig/target_false_negatives'  # 指定目标文件夹
#     os.makedirs(target_false_negatives_dir, exist_ok=True)
#     copy_images(datadir, target_false_negatives_dir, false_negatives)


    
            
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

    flag = torch.cuda.is_available()
    if flag:
        print("CUDA可使用")
    else:
        print("CUDA不可用")

    ngpu= 1
#     os.environ['CUDA_VISIBLE_DEVICES'] ='0'
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    torch.cuda.set_device(0)
    print("驱动为：",device)
    print("GPU型号: ",torch.cuda.get_device_name(0))


    img_data_dir = 'pd'
    skin_type = 'dermatology_images'
    losses_choice = 'Contrastive_loss'
    model_select = 'resnet_18'
    dataset_choice = 'pd'
    category = 'your_category'  # 设置正确的类别值
    n_epoch = 300
    n_classes = 3

    train_eval(img_data_dir, skin_type, losses_choice, model_select, dataset_choice, category, n_epoch, n_classes)
#     train_eval(args.img_data_dir, args.skin_type, args.losses_choice, args.model_select, args.dataset_choice,args.category, args.n_epoch, args.n_classes)


if __name__ == '__main__':
    main()

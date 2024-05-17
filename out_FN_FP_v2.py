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
# import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve, auc, accuracy_score, precision_recall_fscore_support, jaccard_score

import shutil
from sklearn.model_selection import StratifiedKFold
from skimage import data_dir,io,color
from utils import *

import numpy as np
from PIL import Image

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_images_on_scatter(ax, points, image_paths, num_images, exp_mode, zoom=0.5):
    # Ensure num_images does not exceed the length of image_paths or points
    num_images = min(num_images, len(image_paths), len(points))
    
    for point, image_path in zip(points[:num_images], image_paths[:num_images]):
        folder = f'/home/feng/jeding/PD_contrastive_research_0817/{exp_mode}_saved_fig/'
        img = Image.open(folder + image_path)
        img.thumbnail((80, 80), Image.ANTIALIAS)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, point, frameon=False)
        ax.add_artist(ab)


def save_test_images(images, metadata, path):
    # 轉換 metadata 為 DataFrame
    df_metadata = pd.DataFrame(metadata)
    
    for idx, image_tensor in enumerate(images):
        
        # 獲取當前圖像的 image_id
        image_id = df_metadata[0][idx].astype(int).item()  # 假設每個 image 只有一個對應的 image_id
        # import pdb;pdb.set_trace()
        # 將圖像數據從 Tensor 轉換為 NumPy 並調整通道順序
        image = Image.fromarray((image_tensor.numpy() * 255).astype('uint8').transpose(2, 1, 0))

        # 儲存圖像
        image.save(f"{path}/00{image_id}.jpg")
        # import pdb;pdb.set_trace()
def copy_images(source_dir, target_dir, indices):
    """Copy images from the source directory to the target directory based on provided indices."""
    # 獲取源資料夾中所有檔案的列表，假設只有圖片
    files = os.listdir(source_dir)
    files.sort()  # 確保檔案是按字典順序排序的，這通常是必需的以保持一致性

    # 遍歷提供的索引列表
    for idx in indices:
        if idx - 1 < len(files):  # 確保索引不會超出列表範圍
            image_name = files[idx - 1]  # 獲取對應的檔案名
            image_path = os.path.join(source_dir, image_name)
            target_path = os.path.join(target_dir, image_name)
            shutil.copy(image_path, target_path)  # 複製檔案
        else:
            print(f"Index {idx} is out of bounds.")  # 索引超出範圍時打印錯誤訊息


def dataloader(datadir,skin_type, exp_mode):
    

    path_img = '/home/feng/jeding/PD_contrastive_research_0817/spect_513_data'+ '/'
    path_meta = '/home/feng/jeding/PD_contrastive_research_0817/spect_513_data' + '/' 
    coll = io.ImageCollection('/home/feng/jeding/PD_contrastive_research_0817/spect_513_data/spect_img_a2/*.jpg')

    #coll = io.ImageCollection(r'C:\Users\adm\SPECT_3_3\mask_three\*.jpg')
    #coll = io.ImageCollection(r'C:\Users\adm\SPECT_3_3\all_SPECT_RGB\*.jpg')
    raw_image = io.concatenate_images(coll)


    raw_meta = pd.read_csv(path_meta  + 'label_513.csv') 
    label_630_id = raw_meta[raw_meta['ID'] < 634]
    raw_data = raw_image[label_630_id['Python_ID']] / 255

    label_3 = label_630_id['Lebel_3'].values
    label_2 = label_630_id['Label_2'].values

    _, raw_data = transform_label(raw_data, label_2 ,label_3, exp_mode)
    train_length = int(len(raw_data)*0.8)


    raw_data = torch.tensor(raw_data).transpose(1,3)
    # raw_data = torch.flatten(raw_data, start_dim=1)
    raw_data = np.array(raw_data)
    _, raw_patients_feature_412 = transform_label(np.asarray(label_630_id.iloc[:,8:20]),label_2 ,label_3, exp_mode)
    _, raw_patients_feature_412_df = transform_label_output_df(label_630_id,label_2 ,label_3, exp_mode)
    # raw_patients_feature_412 = np.asarray(label_630_id.iloc[:,8:20])
    # import pdb;pdb.set_trace()

    raw_image_train = raw_data[0:train_length]
    raw_image_test = raw_data[train_length:]
    
    # import pdb;pdb.set_trace()
    path_prefix = f'/home/feng/jeding/PD_contrastive_research_0817/{exp_mode}_saved_fig'
    raw_image_test_save = torch.from_numpy(raw_image_test)
# 假設 raw_image_test 已經是一個 NumPy 數組並且準備好用於儲存
    # import pdb;pdb.set_trace()
    save_test_images(raw_image_test_save, raw_patients_feature_412_df[train_length:], path_prefix)
    # raw_image_train = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/train_clinic_f_413.npy') /255
    # raw_image_test = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/test_clinic_f_395.npy') /255
    # import pdb;pdb.set_trace()

    raw_f_train = raw_patients_feature_412[0:train_length]
    raw_f_test = raw_patients_feature_412[train_length:]
    # raw_f_train = preprocessing.scale(raw_f_train )
    # raw_f_test = preprocessing.scale(raw_f_test )

    image_data_train = raw_image_train 
    feature_data_train = raw_f_train 


    image_data_train = torch.from_numpy(image_data_train ).float() #torch.Size([413, 128, 128, 3])


    feature_data_train = torch.from_numpy(feature_data_train).float()

    image_data_flatten = torch.flatten(image_data_train, start_dim=1)

    # import pdb;pdb.set_trace()
    adj_train_img = build_knn_graph(image_data_flatten,200).float()

    # adj_train_img = kneighbors_graph(np.array(image_data_flatten), 200, mode='connectivity', include_self=True).toarray()



    # adj_train_img = torch.from_numpy(adj_train_img).float()


    image_data_test = torch.from_numpy(raw_image_test ).float()

    data_features_test = raw_f_test 
    test_feature_data = torch.from_numpy(data_features_test).float()

# 创建测试用的邻接矩阵（这里假设所有病人之间都有连接）
# test_adjacency_matrix = torch.ones((100, 100))


## testing image adj
    image_data_test_flatten = torch.flatten(image_data_test, start_dim=1)
    # image_data_test_flatten = image_data_test

    # adj_test_img = kneighbors_graph(np.array(image_data_test_flatten), 200, mode='connectivity', include_self=True).toarray()

    adj_test_img = build_knn_graph(image_data_test_flatten, 40).float() #[104, 104]


    adj_f_knn_train =  build_knn_graph(raw_f_train, 200).float()

    # adj_f_knn_train = adj_f_knn_train.toarray()
    # adj_f_knn_train = torch.from_numpy(adj_f_knn_train).float()
    # adj_f_knn_test = kneighbors_graph(np.array(raw_f_test), 300, mode='connectivity', include_self=True)
    adj_f_knn_test = build_knn_graph(raw_f_test, 40).float()
        
    return image_data_train, feature_data_train, adj_train_img, adj_f_knn_train, image_data_test, test_feature_data, adj_test_img, adj_f_knn_test

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from sklearn.cluster import KMeans
from keras.applications.mobilenet import MobileNet, preprocess_input
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
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
    
class Model_PD(nn.Module):
    def __init__(self, projection, input_resnet, n_classes):
        super(Model_PD, self).__init__()
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
        self.linear4 = nn.Linear(256,2)
        self.linear_f = nn.Linear(18,2)
        self.attention_f = AttentionNetwork_f(15)
        self.attention_img = AttentionNetwork_img(1027)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, x_f,adjacency_img, adjacency_f):
        # import pdb;pdb.set_trace()
        # x = self.encoder(x) #torch.Size([300, 65536])
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # x_encoder = self.cnn_encoder(x)
        
        x_encoder = self.input_resnet(x)
        # x_encoder = self.cnn_encoder(x)
       
#         x_encoder = self.cnn_encoder(x)

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
       
        
        w_att_f = self.attention_f(x_f_fusion) #MLP
        w_att_img = self.attention_img(x_img) #MLP
        
        attended_gat_f = x_gat_f * w_att_f
        attended_gat_img = x_gat_img * w_att_img
        x_f_att = torch.cat((x_f_fusion, attended_gat_f), 1) # torch.Size([300, 17])
        x_img_att = torch.cat((x_img, attended_gat_img), 1) # torch.Size([300, 1029])
        # 這部份Journal可以寫數學式子w
        # import pdb;pdb.set_trace()
        emb1  = self.linear_f(x_f_att)
        # emb1  = self.linear4(emb1)
        
        emb2  = self.linear3(x_img_att)
        emb2  = self.linear4(emb2)
        
        # x1 = self.linear1(x_f)
        

        # # x2 = self.linear2(x_encoder)
        # x2 = self.linear1(x_img)

        # x1 = self.softmax(x1)
        # x2 = self.softmax(x2)

        # x1 = self.linear1(attended_gcn)
        z = emb1 + emb2

        
        #import pdb;pdb.set_trace()
       
        #emb2 =  torch.from_numpy(emb2).to(torch.float32).to(device)
        #import pdb;pdb.set_trace()

    
        #z = torch.from_numpy(z).to(device)
        #emb2 = torch.from_numpy(emb2).float().to(device)
        
    

        #x1 = self.softmax(emb1) 
        #x2 = self.softmax(emb2) 
        
        #import pdb;pdb.set_trace()
    
        z  = self.softmax(z).to(torch.float32)
        
        # import pdb;pdb.set_trace()

        return emb1, emb2, z
    
    def nonlinear_transformation(self, h):
        z = self.projection(h)
        return z


import json


def train_eval(datadir,skin_type, loss_select, model_select , dataset_choice ,category, epoch, n_classes, exp_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    torch.cuda.is_available()
    n_classes = 2
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
    # num_features = model_net.classifier.in_features # Desnet101
    # import pdb;pdb.set_trace()
    model_net.fc = nn.Linear(num_features, num_classes) #512 # Resnet
    model_net.fc = model_net.fc.to(device) #512 # Resnet
    # model_net.classifier = nn.Linear(num_features, num_classes) #desnet
    datadir = 'pd'
    image_data_train, feature_data_train, adj_train_img, adj_f_knn_train, image_data_test, test_feature_data, adj_test_img, adj_f_knn_test = dataloader(datadir,skin_type, exp_mode)
    

    
    projection = Projection(262, 3)
    if datadir == 'skin':
        model = Model_SKIN(projection, model_net, n_classes).to(device)
    elif datadir == 'abide':
        model = Model_ABIDE(projection, model_net, n_classes).to(device)
    elif datadir == 'pd':
        model = Model_PD(projection, model_net, n_classes).to(device)

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
    class_name = category
    n_epochs = epoch

    training_range = tqdm(range(n_epochs))

    for epoch in training_range:
        optimizer.zero_grad()
        # cnn_z  =  cnn_encoder(image_data)
        # 前向传播
        
        image_data_train = image_data_train.to(device)
        feature_data_train = feature_data_train.to(device)
        adj_train_img = adj_train_img.to(device)
        adj_f_knn_train = adj_f_knn_train.to(device)
        # import pdb;pdb.set_trace()
        output1, output2, emb = model(image_data_train , feature_data_train,adj_train_img, adj_f_knn_train)
         
        y = torch.tensor(label_return(dataset_choice, class_name, "train", exp_mode)).to(device)
        # import pdb;pdb.set_trace()
        loss_ce1 = criterion1(output1, y)
        loss_ce2 = criterion1(output2, y)
        alpha = 0.4

        if loss_select == 'Contrastive_loss':
            adj = adj_train_img +  adj_f_knn_train
            diag = torch.diag(adj.sum(dim=1))
            loss_extra = criterion2( emb, adj_train_img, adj_f_knn_train, y, output1, output2, diag).to(device)
            loss = (1-alpha)*(loss_ce1 + loss_ce2) + alpha* loss_extra
            # loss = loss_ce1 + loss_ce2
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
            # loss = loss_ce1 + loss_ce2
            loss = loss_ce1


        loss.backward()
        optimizer.step()
        # torch.save(model,f'{skin_type}_{epoch}epoch_save.pt')
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))

    model.eval()
    with torch.no_grad():
        # import pdb;pdb.set_trace()
        image_data_test = image_data_test.to(device)
        test_feature_data = test_feature_data.to(device)
        adj_test_img = adj_test_img.to(device)
        adj_f_knn_test = adj_f_knn_test.to(device)
        
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
        #pred = cluster_labels
        # test_output = test_output.argmax(dim=1)
        
        
        # y_test = torch.empty(100).random_(2)
    #     y_test = torch.tensor(label_3_test).to(device)
        y_test = torch.from_numpy(label_return(dataset_choice ,class_name, "test", exp_mode)).to(device)

        # import pdb;pdb.set_trace()
        correct = (pred  == y_test).sum().item()
        accuracy = correct / len(y_test)
        
        
        # import pdb;pdb.set_trace()
        print(f"++++Use {model_select} model+++")
        print(calculate_metrics_new(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
        print("Loss:", loss_select, "class_name",class_name,"Accuracy:", accuracy)
        print(classification_report(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))

        # import pdb;pdb.set_trace()
        TN = (pred ==  y_test).nonzero(as_tuple=True)[0].cpu().numpy()
        # false_negatives = (pred < y_test).nonzero(as_tuple=True)[0].cpu().numpy()
        # import pdb;pdb.set_trace()

        # # Save results to JSON
        # result_data = {'FalsePositives': false_positives.tolist(), 'FalseNegatives': false_negatives.tolist()}

        # with open('false_samples.json', 'w') as json_file:
        #     json.dump(result_data, json_file, indent=4)

        # Prepare directories for saving false classified images
        # import pdb;pdb.set_trace()
        # target_false_positives_dir = '/home/feng/jeding/PD_contrastive_research_0817/test_out_results/target_false_positives'
        # os.makedirs(target_false_positives_dir, exist_ok=True)
        source_dif = f'/home/feng/jeding/PD_contrastive_research_0817/{exp_mode}_saved_fig'
        # copy_images(source_dif, target_false_positives_dir, false_positives)
        files = os.listdir(source_dif)
        # import pdb;pdb.set_trace()
        # files.sort()  # 確保檔案是按字典順序排序的，這通常是必需的以保持一致性
        
        # 遍歷提供的索引列表
        final_index = []
        TN_index = []
        for idx in TN:
            if idx - 1 < len(files):  # 確保索引不會超出列表範圍
                image_name = files[idx - 1]  # 獲取對應的檔案名
                final_index.append(image_name)
                TN_index.append(idx)
        # import pdb;pdb.set_trace()
                # image_path = os.path.join(source_dir, image_name)

        # target_false_negatives_dir = '/home/feng/jeding/PD_contrastive_research_0817/test_out_results/target_false_negatives'
        # os.makedirs(target_false_negatives_dir, exist_ok=True)
        # copy_images(source_dif, target_false_negatives_dir, false_negatives)

        # cm = confusion_matrix(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy())
        # print("Confusion Matrix:")
        # print(cm)

        # 可视化混淆矩阵并保存为图像
        # plt.figure(figsize=(10,7))  # 可以调整图形的大小以更好地适应字体大小的增加
        # sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', annot_kws={"size": 30},
        #             xticklabels=['Mid-abnormal', 'Abnormal'],  # 自定义X轴标签
        #             yticklabels=['Mid-abnormal', 'Abnormal'])  # 自定义Y轴标签)  # 增加格内数字大小
        # plt.title('Confusion Matrix', fontsize=20)  # 增加标题大小
        # plt.ylabel('True Label', fontsize=20)  # 增加y轴标签大小
        # plt.xlabel('Predicted Label', fontsize=20)  # 增加x轴标签大小
        # plt.xticks(fontsize=20)  # 增加x轴刻度大小
        # plt.yticks(fontsize=20)  # 增加y轴刻度大小
        # plt.savefig(f'{exp_mode}_confusion_matrix.png')  # 保存图像
        # plt.close()

        test_output_np = test_output.cpu().detach().numpy()
        y_test_np = y_test.cpu().detach().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        test_output_2d = tsne.fit_transform(test_output_np)
        image_indices = final_index
        # import pdb;pdb.set_trace()
        image_paths = [os.path.join(source_dif, f"{idx}") for idx in image_indices]
        plt.figure(figsize=(12, 8))
        # scatter = plt.scatter(test_output_2d[:, 0], test_output_2d[:, 1], c=y_test_np, cmap='viridis', alpha=0.6)
        # plt.colorbar(scatter, ticks=range(len(set(y_test_np))))

        # image_indices_ = TN_index 
        # # import pdb;pdb.set_trace()
        # plot_images_on_scatter(plt.gca(), test_output_2d[image_indices_], image_indices, 10)
        colors = ['orange' if label == 0 else 'red' for label in y_test_np]
        markers = ['o' if label == 0 else '^' for label in y_test_np]
        # import pdb;pdb.set_trace()
# 繪製散點圖
        for color, marker, (x, y) in zip(colors, markers, test_output_2d):
            plt.scatter(x, y, color=color, marker=marker, alpha=0.6, s=150)

        # 添加圖例
        import matplotlib.lines as mlines
        normal_legend = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markeredgecolor='black',markersize=20, label='Mid-abnormal')
        abnormal_legend = mlines.Line2D([], [], color='red', marker='^', linestyle='None', markeredgecolor='black',markersize=20, label='Abnormal')
        plt.legend(handles=[normal_legend, abnormal_legend], fontsize=15)

        # 繪製圖片在散點圖上
        image_indices_ = TN_index
        plot_images_on_scatter(plt.gca(), test_output_2d[image_indices_], image_indices, 7, exp_mode)


        plt.title(f'Mid-abnormal vs Abnormal', fontsize=25)
        plt.xlabel('t-SNE dimension 1', fontsize=20)
        plt.ylabel('t-SNE dimension 2', fontsize=20)
        plt.savefig(f'{exp_mode}_tsne_scatter.png')
        plt.close()




    
            
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
    exp_mode='mid_abnormal'
    n_epoch = 300
    n_classes = 2

    train_eval(img_data_dir, skin_type, losses_choice, model_select, dataset_choice, category, n_epoch, n_classes, exp_mode)
#     train_eval(args.img_data_dir, args.skin_type, args.losses_choice, args.model_select, args.dataset_choice,args.category, args.n_epoch, args.n_classes)


if __name__ == '__main__':
    main()

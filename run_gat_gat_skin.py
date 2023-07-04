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
from skimage import data_dir,io,color
from torch.autograd import Variable
from sklearn.metrics import classification_report
from sklearn import preprocessing
from utils import build_knn_graph, label_return, plot_ROC, calculate_metrics_new
from model import Projection, Model, CNN
from losses import WeightedCrossEntropyLoss, contrastive_loss, info_loss, MGECLoss, SACLoss
# device = torch.device("mps" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")



resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet = resnet.to(device)
# 将最后一层的输出维度修改为类别数目
num_classes = 1024
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)
resnet.fc = resnet.fc.to(device)

raw_image_train = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/dermatology_images/train_derm_img_413.npy') 

raw_image_test = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/dermatology_images/test_derm_img_395.npy') 

# raw_image_train = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/train_clinic_f_413.npy') /255
# raw_image_test = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/test_clinic_f_395.npy') /255


raw_f_train = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/meta_ok/meta_train_413.npy')
raw_f_test = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/meta_ok/meta_test_395.npy')
raw_f_train = preprocessing.scale(raw_f_train )
raw_f_test = preprocessing.scale(raw_f_test )

image_data_train = raw_image_train 
feature_data_train = raw_f_train 

# 转换为PyTorch张量
image_data_train = torch.from_numpy(image_data_train ).float()
image_data_train = torch.tensor(image_data_train ).transpose(1,3)
feature_data_train = torch.from_numpy(feature_data_train).float()
# import pdb;pdb.set_trace()
image_data_flatten = torch.flatten(image_data_train, start_dim=1)
image_data_flatten = image_data_flatten 

# adj_train_img = kneighbors_graph(np.array(image_data_flatten), 200, mode='connectivity', include_self=True).toarray()

adj_train_img = build_knn_graph(image_data_flatten,300).float()
# import pdb;pdb.set_trace()
# adj_train_img = torch.from_numpy(adj_train_img).float()


image_data_test = torch.from_numpy(raw_image_test ).float()
image_data_test = torch.tensor(image_data_test ).transpose(1,3)
data_features_test = raw_f_test 
test_feature_data = torch.from_numpy(data_features_test).float()

# 创建测试用的邻接矩阵（这里假设所有病人之间都有连接）
# test_adjacency_matrix = torch.ones((100, 100))


## testing image adj

image_data_test_flatten = torch.flatten(image_data_test, start_dim=1)
# image_data_test_flatten = image_data_test
# adj_test_img = kneighbors_graph(np.array(image_data_test_flatten), 200, mode='connectivity', include_self=True).toarray()

adj_test_img = build_knn_graph(image_data_test_flatten, 200).float()


adj_f_knn_train =  build_knn_graph(raw_f_train, 300).float()
# adj_f_knn_train = adj_f_knn_train.toarray()
# adj_f_knn_train = torch.from_numpy(adj_f_knn_train).float()
# adj_f_knn_test = kneighbors_graph(np.array(raw_f_test), 300, mode='connectivity', include_self=True)
adj_f_knn_test = build_knn_graph(raw_f_test, 300).float()

projection = Projection(262, 3)
model = Model(projection, resnet).to(device)
cnn_encoder = CNN().to(device)

class_weights = torch.tensor([0.5, 0.5,0.5])  # 自定義的類別權重
criterion5 = WeightedCrossEntropyLoss(weight=class_weights)
criterion2 = contrastive_loss
# criterion3 = SACLoss()
criterion3 = MGECLoss()
# criterion3 = info_loss

# criterion3 = loss_dependence
optimizer = optim.Adam(model.parameters(), lr=0.001)




class_name = "PIG"

n_epochs = 300
training_range = tqdm(range(n_epochs))

for epoch in training_range:
    optimizer.zero_grad()
    # cnn_z  =  cnn_encoder(image_data)
    # 前向传播
    
    image_data_train = image_data_train.to(device)
    feature_data_train = feature_data_train.to(device)
    adj_train_img = adj_train_img.to(device)
    adj_f_knn_train = adj_f_knn_train.to(device)
    
    
    output1, output2, emb = model(image_data_train , feature_data_train,adj_train_img, adj_f_knn_train)

    # import pdb;pdb.set_trace()
    # x_f_att_z = model.nonlinear_transformation(x_f_att)
    # x_img_att_z = model.nonlinear_transformation( x_img_att)
    # output1  = model(image_data, adjacency_matrix ,adj_train_img)
    # import pdb;pdb.set_trace()
    # output = model(image_data, adjacency_matrix)
    # model.nonlinear_transformation
    # 计算损失
#     y = torch.tensor(label_3_300_train).to(device)
    y = torch.tensor(label_return(class_name, "train")).to(device)
    # import pdb;pdb.set_trace()
    # loss = criterion(output, feature_data)
    # loss = criterion(output, y, 0.03)
    # import pdb;pdb.set_trace()
    # loss = criterion(output, y)
    # loss1 = criterion1(output1, y)
    # loss2 = criterion1(output2, y)
    # loss3 = criterion2(x_f_att_z, x_img_att_z, adjacency_matrix ,adj_train_img, y)
#     loss3 = criterion2( emb, adj_train_img, adj_f_knn_train, y)
    loss1 = criterion5(output1, y)
    loss2 = criterion5(output2, y)
    
    adj = adj_train_img +  adj_f_knn_train
    diag = torch.diag(adj.sum(dim=1))

    loss4 = criterion3(output1, output2, adj, diag )
#     loss4 = criterion3(emb, adj_train_img, adj_f_knn_train, y)
    
    

    # loss3 = criterion2( output1, output2, y)
    
    # loss4 = criterion4(emb, y)

#     loss = loss1 + loss2

    alpha = 0.4
#     loss = (1-alpha)*(loss1 + loss2) + alpha* loss3
    loss = (1-alpha)*(loss1 + loss2) + alpha* loss4
    # loss = (1-alpha)*loss1  + alpha * loss2
    
#     loss = (1-alpha)*(loss1 + loss2) + alpha* loss4
#     loss = loss1 + loss2 + loss3 + loss4

    
    # import pdb;pdb.set_trace()
    # 反向传播和优化
    # loss.requires_grad_(True)
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))

model.eval()
with torch.no_grad():
    # import pdb;pdb.set_trace()
    image_data_test = image_data_test.to(device)
    test_feature_data = test_feature_data.to(device)
    adj_test_img = adj_test_img.to(device)
    adj_f_knn_test = adj_f_knn_test.to(device)
    
    test_output1, test_output2, emb  = model(image_data_test, test_feature_data , adj_test_img, adj_f_knn_test  )

    # test_output1  = model(test_image_data, test_adjacency_matrix, adj_test_img )
    m = nn.Softmax(dim=1)
    # import pdb;pdb.set_trace()
    test_output = test_output1 + test_output2
#     test_output = emb

    pred =  m(test_output).argmax(dim=1)
    # test_output = test_output.argmax(dim=1)
    
    
    # y_test = torch.empty(100).random_(2)
#     y_test = torch.tensor(label_3_test).to(device)
    y_test = torch.from_numpy(label_return(class_name, "test")).to(device)

    
    correct = (pred  == y_test).sum().item()
    accuracy = correct / len(y_test)
    print("class_name",class_name,"Accuracy:", accuracy)
    # import pdb;pdb.set_trace()
    print(calculate_metrics_new(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
    print(classification_report(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
    
    plot_ROC(pred.cpu().detach().numpy() , y_test.cpu().detach().numpy(), 3)

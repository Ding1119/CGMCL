import numpy as np
import torch
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

def build_knn_graph(input_data, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(input_data)
    _, indices = knn.kneighbors(input_data)
    adjacency_matrix = torch.zeros(input_data.shape[0], input_data.shape[0])
    for i, neighbors in enumerate(indices):
        adjacency_matrix[i, neighbors] = 1
        
    return adjacency_matrix

def dataloader_breast(task_type, img_data, clinical_data):
    
    if task_type == "train":

        raw_f_train = clinical_data
        
        # raw_f_train = preprocessing.scale(raw_f_train )
        image_data_train = img_data.float()
        # image_data_train = image_data_train.reshape(300, 3,128,128)
        image_data_train = img_data.view(30,-1)
        image_data_train_flatten = torch.flatten(image_data_train.squeeze(dim=0), start_dim=1)
        feature_data_train = raw_f_train 
        
        # 转换为PyTorch张量

        # image_data_train = torch.tensor(image_data_train ).transpose(1,3)
        feature_data_train = feature_data_train.float()
        adj_train_img = build_knn_graph(image_data_train_flatten,15).float()
        
        feature_data_train = feature_data_train.repeat(1, 6).permute(1,0)

        adj_f_knn_train =  feature_data_train @ feature_data_train.T
        
        return image_data_train, feature_data_train, adj_train_img, adj_f_knn_train

    else:
        raw_f_test = clinical_data
        
        # raw_f_train = preprocessing.scale(raw_f_train )
        image_data_test = img_data.float()
        # image_data_train = image_data_train.reshape(300, 3,128,128)
        image_data_test = img_data.view(30,-1)
        image_data_test_flatten = torch.flatten(image_data_test.squeeze(dim=0), start_dim=1)
        feature_data_test = raw_f_test
        
        # 转换为PyTorch张量

        # image_data_train = torch.tensor(image_data_train ).transpose(1,3)
        feature_data_test = feature_data_test.float()
        adj_test_img = build_knn_graph(image_data_test_flatten,15).float()
        
        feature_data_test = feature_data_test.repeat(1, 6).permute(1,0)

        adj_f_knn_test =  feature_data_test @ feature_data_test.T

    return image_data_test, feature_data_test, adj_test_img, adj_f_knn_test

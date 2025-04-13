import numpy as np
import torch
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from utils import *


def build_knn_graph(input_data, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(input_data)
    _, indices = knn.kneighbors(input_data)
    adjacency_matrix = torch.zeros(input_data.shape[0], input_data.shape[0])
    for i, neighbors in enumerate(indices):
        adjacency_matrix[i, neighbors] = 1
        
    return adjacency_matrix

def dataloader(datadir,skin_type, exp_mode):

    if datadir == 'skin':

        path_img = './dataset_skin/' + skin_type + '/'
        raw_image_train = np.load(path_img  + 'train_derm_img_413.npy') 
        raw_image_test = np.load(path_img  + 'test_derm_img_395.npy') 
        
        raw_f_train = np.load('./dataset_skin/meta_features/'+ 'meta_train_413.npy')
        raw_f_test = np.load('./dataset_skin/meta_features/' +'meta_test_395.npy')

        raw_f_train = preprocessing.scale(raw_f_train )
        raw_f_test = preprocessing.scale(raw_f_test )
        image_data_train = raw_image_train 
        feature_data_train = raw_f_train 
     
        image_data_train = torch.from_numpy(image_data_train ).float()
        image_data_train = torch.tensor(image_data_train ).transpose(1,3)
        feature_data_train = torch.from_numpy(feature_data_train).float()

        image_data_flatten = torch.flatten(image_data_train, start_dim=1)
        image_data_flatten = image_data_flatten 
        adj_train_img = build_knn_graph(image_data_flatten,300).float()


        image_data_test = torch.from_numpy(raw_image_test ).float()
        image_data_test = torch.tensor(image_data_test ).transpose(1,3)
        data_features_test = raw_f_test 
        test_feature_data = torch.from_numpy(data_features_test).float()

        image_data_test_flatten = torch.flatten(image_data_test, start_dim=1)

        adj_test_img = build_knn_graph(image_data_test_flatten, 200).float()


        adj_f_knn_train =  build_knn_graph(raw_f_train, 300).float()
        adj_f_knn_test = build_knn_graph(raw_f_test, 300).float()
       
    return image_data_train, feature_data_train, adj_train_img, adj_f_knn_train, image_data_test, test_feature_data, adj_test_img, adj_f_knn_test



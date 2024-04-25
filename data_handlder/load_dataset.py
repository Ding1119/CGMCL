import numpy as np
import torch
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from skimage import data_dir,io,color
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

        path_img = '/home/jding/Documents/PD_contrastive_research_0817/skin_dataset_ok/' + skin_type + '/'
        raw_image_train = np.load(path_img  + 'train_clinic_img_413.npy') 
        # import pdb;pdb.set_trace()
        raw_image_test = np.load(path_img  + 'test_clinic_img_395.npy') 
        
        # raw_image_train = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/train_clinic_f_413.npy') /255
        # raw_image_test = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/test_clinic_f_395.npy') /255

        
        raw_f_train = np.load('/home/jding/Documents/PD_contrastive_research_0817/skin_dataset_ok_old/meta_ok/'+ 'meta_train_413.npy')
        raw_f_test = np.load('/home/jding/Documents/PD_contrastive_research_0817/skin_dataset_ok_old/meta_ok/' +'meta_test_395.npy')
        raw_f_train = preprocessing.scale(raw_f_train )
        raw_f_test = preprocessing.scale(raw_f_test )
        image_data_train = raw_image_train 
        feature_data_train = raw_f_train 
     
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

    elif datadir == 'pd':
        path_img = '/home/jding/Documents/PD_contrastive_research_0817/spect_513_data'+ '/'
        path_meta = '/home/jding/Documents/PD_contrastive_research_0817/spect_513_data' + '/' 
        coll = io.ImageCollection('/home/jding/Documents/PD_contrastive_research_0817/spect_513_data/spect_img_a2/*.jpg')

        #coll = io.ImageCollection(r'C:\Users\adm\SPECT_3_3\mask_three\*.jpg')
        #coll = io.ImageCollection(r'C:\Users\adm\SPECT_3_3\all_SPECT_RGB\*.jpg')
        raw_image = io.concatenate_images(coll)

        
        raw_meta = pd.read_csv(path_meta  + 'label_513.csv') 
        label_630_id = raw_meta[raw_meta['ID'] < 634]
        raw_data = raw_image[label_630_id['Python_ID']] / 255

        label_3 = label_630_id['Lebel_3'].values
        label_2 = label_630_id['Label_2'].values
        
        _, raw_data = transform_label(raw_data, label_3, exp_mode='normal_mid')
        train_length = int(len(raw_data)*0.8)


        raw_data = torch.tensor(raw_data).transpose(1,3)
        # raw_data = torch.flatten(raw_data, start_dim=1)
        raw_data = np.array(raw_data)
        _, raw_patients_feature_412 = transform_label(np.asarray(label_630_id.iloc[:,8:20]), label_3, exp_mode='normal_mid')
        # raw_patients_feature_412 = np.asarray(label_630_id.iloc[:,8:20])
        # import pdb;pdb.set_trace()
        
        raw_image_train = raw_data[0:train_length]
        raw_image_test = raw_data[train_length:]
        # import pdb;pdb.set_trace()
        # raw_image_train = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/train_clinic_f_413.npy') /255
        # raw_image_test = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/test_clinic_f_395.npy') /255
        

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
        # import pdb;pdb.set_trace()


    elif datadir == 'abide':
        path_img = '/home/feng/jeding/PD_contrastive_research_0817/data_storage'+ '/'
        path_meta = '/home/feng/jeding/PD_contrastive_research_0817/data_storage' + '/' 
        # import pdb;pdb.set_trace()
        raw_image_train = np.load(path_img  + 'X_train.npy') 

        raw_image_test = np.load(path_img  + 'X_test.npy') 
        
        # raw_image_train = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/train_clinic_f_413.npy') /255
        # raw_image_test = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/test_clinic_f_395.npy') /255
        

        raw_f_train = np.load(path_meta + 'X_train_f.npy')
        raw_f_test = np.load(path_meta  +'X_test_f.npy')
        # raw_f_train = preprocessing.scale(raw_f_train )
        # raw_f_test = preprocessing.scale(raw_f_test )
        
        image_data_train = raw_image_train 
        feature_data_train = raw_f_train 

        
        image_data_train = torch.from_numpy(image_data_train ).float() #torch.Size([413, 128, 128, 3])
        
    
        feature_data_train = torch.from_numpy(feature_data_train).float()
        
        # image_data_flatten = torch.flatten(image_data_train, start_dim=1)
        image_data_flatten = image_data_train
        adj_train_img = build_knn_graph(image_data_flatten,500).float()
        
        # adj_train_img = kneighbors_graph(np.array(image_data_flatten), 200, mode='connectivity', include_self=True).toarray()

        
        # import pdb;pdb.set_trace()
        # adj_train_img = torch.from_numpy(adj_train_img).float()


        image_data_test = torch.from_numpy(raw_image_test ).float()
    
        data_features_test = raw_f_test 
        test_feature_data = torch.from_numpy(data_features_test).float()

    # 创建测试用的邻接矩阵（这里假设所有病人之间都有连接）
    # test_adjacency_matrix = torch.ones((100, 100))


    ## testing image adj

        image_data_test_flatten = image_data_test
        # image_data_test_flatten = image_data_test
        # adj_test_img = kneighbors_graph(np.array(image_data_test_flatten), 200, mode='connectivity', include_self=True).toarray()
        
        adj_test_img = build_knn_graph(image_data_test_flatten, 80).float() #[104, 104]

        
        adj_f_knn_train =  build_knn_graph(raw_f_train, 500).float()
        
        # adj_f_knn_train = adj_f_knn_train.toarray()
        # adj_f_knn_train = torch.from_numpy(adj_f_knn_train).float()
        # adj_f_knn_test = kneighbors_graph(np.array(raw_f_test), 300, mode='connectivity', include_self=True)
        adj_f_knn_test = build_knn_graph(raw_f_test, 80).float()
        
       
    return image_data_train, feature_data_train, adj_train_img, adj_f_knn_train, image_data_test, test_feature_data, adj_test_img, adj_f_knn_test



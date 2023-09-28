import torch
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from torch_geometric.data import Data

def label_return(dataset_choice, category,label):
    if dataset_choice == 'skin':
        if label == 'train':
            
            raw_train_label = pd.read_csv('/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/train_labels_df_413.csv') 
            
            return np.array(raw_train_label[f'{category}'])
        else:

            raw_test_label = pd.read_csv('/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/test_labels_df_395.csv')
            
            return np.array(raw_test_label[f'{category}'])
        
    elif dataset_choice == 'abide':
        if label == 'train':

            raw_train_label = np.load('/home/feng/jeding/PD_contrastive_research_0817/data_storage/y_train.npy') 
            
            return raw_train_label
        else:
            raw_test_label = np.load('/home/feng/jeding/PD_contrastive_research_0817/data_storage/y_test.npy') 
            
            return raw_test_label

def build_knn_graph(input_data, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(input_data)
    _, indices = knn.kneighbors(input_data)
    adjacency_matrix = torch.zeros(input_data.shape[0], input_data.shape[0])
    for i, neighbors in enumerate(indices):
        adjacency_matrix[i, neighbors] = 1
        
    return adjacency_matrix

def dataloader(datadir,skin_type):

    if datadir == 'skin':

        path_img = '/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/' + skin_type + '/'
        raw_image_train = np.load(path_img  + 'train_derm_f_413.npy') 

        raw_image_test = np.load(path_img  + 'test_derm_f_395.npy') 
        
        # raw_image_train = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/train_clinic_f_413.npy') /255
        # raw_image_test = np.load('/Users/test/Documents/Contrastive_PD/skin_dataset_ok/clinical_images/test_clinic_f_395.npy') /255

        
        raw_f_train = np.load('/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/meta_ok/'+ 'meta_train_413.npy')
        raw_f_test = np.load('/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/meta_ok/' +'meta_test_395.npy')
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
        raw_f_train = preprocessing.scale(raw_f_train )
        raw_f_test = preprocessing.scale(raw_f_test )
        
        image_data_train = raw_image_train 
        feature_data_train = raw_f_train 

        
        image_data_train = torch.from_numpy(image_data_train ).float() #torch.Size([413, 128, 128, 3])
        
    
        feature_data_train = torch.from_numpy(feature_data_train).float()
        
        # image_data_flatten = torch.flatten(image_data_train, start_dim=1)
        image_data_flatten = image_data_train
        adj_train_img = build_knn_graph(image_data_flatten,300).float()
        
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

        
        adj_f_knn_train =  build_knn_graph(raw_f_train, 300).float()
        
        # adj_f_knn_train = adj_f_knn_train.toarray()
        # adj_f_knn_train = torch.from_numpy(adj_f_knn_train).float()
        # adj_f_knn_test = kneighbors_graph(np.array(raw_f_test), 300, mode='connectivity', include_self=True)
        adj_f_knn_test = build_knn_graph(raw_f_test, 80).float()
        
       
    return image_data_train, feature_data_train, adj_train_img, adj_f_knn_train, image_data_test, test_feature_data, adj_test_img, adj_f_knn_test






def get_y(dataset: [Data]):
    """
    Get the y values from a list of Data objects.
    """
    y = []
    for d in dataset:
        y.append(d['label'])
    return y

class CustomDataset(Dataset):
    def __init__(self, raw_train_paths, raw_test_paths,
                 raw_train_f_paths, raw_test_f_paths,
                  raw_train_label_paths, raw_test_label_paths):
        self.raw_train_paths = raw_train_paths
        self.raw_test_paths = raw_test_paths
        self.raw_train_f_paths = raw_train_f_paths
        self.raw_test_f_paths = raw_test_f_paths
        self.raw_test_paths = raw_test_paths
        self.raw_train_label_paths = raw_train_label_paths
        self.raw_test_label_paths = raw_test_label_paths

        self.data = self.load_and_concatenate(self.raw_train_paths , self.raw_test_paths)
        self.data_f = self.load_and_concatenate(self.raw_train_f_paths , self.raw_test_f_paths)
        self.labels = self.load_and_concatenate_label(self.raw_train_label_paths , self.raw_test_label_paths)
        
    def load_and_concatenate(self, train_path, test_path):
        data = [np.load(train_path),np.load(test_path)]

        # data = [np.load(file_path) for file_path in file_paths]
        return np.concatenate(data, axis=0)
    
    def load_and_concatenate_label(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        category = 'DaG'
        arr_train = np.array(train_df[f'{category}'])
        arr_test = np.array(test_df[f'{category}'])
        
        label = [arr_train, arr_test]

        # data = [np.load(file_path) for file_path in file_paths]
        return np.concatenate(label, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'data_f': self.data[idx],
            'label': self.labels[idx]
        }
        return sample


if __name__ == '__main__':
    # raw_data_path = '/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/'
    # image_data_train, feature_data_train, adj_train_img, adj_f_knn_train, image_data_test, \
    # test_feature_data, adj_test_img, adj_f_knn_test = dataloader('skin', 'dermatology_images')
    # y_train = torch.tensor(label_return('skin','DaG', "train"))
    # y_test = torch.tensor(label_return('skin','DaG', "test"))
    raw_train_path = "/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/dermatology_images/train_derm_f_413.npy"
    raw_test_path = "/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/dermatology_images/test_derm_f_395.npy"
    raw_train_f_path = "/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/clinical_images/train_clinic_f_413.npy"
    raw_test_f_path = "/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/clinical_images/test_clinic_f_395.npy"
    raw_train_label_path = '/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/train_labels_df_413.csv'
    raw_test_label_path = '/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/test_labels_df_395.csv'
    
    dataset = CustomDataset(raw_train_path, raw_test_path, raw_train_f_path, raw_test_f_path,
                                raw_train_label_path,raw_test_label_path)
    
    y = get_y(dataset)
                                
    

    # for _ in range(args.repeat):
    #     seed_everything(random.randint(1, 1000000))  # use random seed for each run
    #     skf = StratifiedKFold(n_splits=args.k_fold_splits, shuffle=True)
    #     for train_index, test_index in skf.split(dataset, y):
    #         train_index = train_index.astype(np.int64)
    #         test_index = test_index.astype(np.int64) 
    import pdb;pdb.set_trace()
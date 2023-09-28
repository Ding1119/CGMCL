from sklearn.model_selection import StratifiedKFold
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
from torch_geometric.loader import DataLoader
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
from utils import build_knn_graph, label_return, plot_auc, print_auc, calculate_metrics_new
from model_resnet_skin import Projection, Model_SKIN, CNN
from model_resnet_abide import Projection, Model_ABIDE, CNN
from data_handlder.train_eval_module import train_and_evaluate
# from losses import WeightedCrossEntropyLoss, contrastive_loss, info_loss, MGECLoss, SACLoss
import argparse

from data_handlder.new_dataloader import CustomDataset, get_y, get_data

import logging
import os


def train_eval(model_select, loss_select):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    raw_train_path = "/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/dermatology_images/train_derm_f_413.npy"
    raw_test_path = "/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/dermatology_images/test_derm_f_395.npy"
    raw_train_f_path = "/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/clinical_images/train_clinic_f_413.npy"
    raw_test_f_path = "/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/clinical_images/test_clinic_f_395.npy"
    raw_train_label_path = '/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/train_labels_df_413.csv'
    raw_test_label_path = '/home/feng/jeding/PD_contrastive_research_0817/skin_dataset_ok/test_labels_df_395.csv'

    if model_select == 'resnet_18':
        model_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # resnet = models.resnet18(pretrained=True)
    elif model_select == 'resnet_34':
         model_net = models.resnet34(pretrained=True)
    elif model_select == 'resnet_50':
        model_net = models.resnet50(pretrained=True)
    elif model_select == 'densenet':
         model_net = models.densenet121(pretrained=True)

    dataset = CustomDataset(raw_train_path, raw_test_path, raw_train_f_path, raw_test_f_path,
                                raw_train_label_path,raw_test_label_path)
    # import pdb;pdb.set_trace()
    y = get_y(dataset)
    data = get_data(dataset)
    # import pdb;pdb.set_trace()
    
    accs, aucs, macros, exp_accs, exp_aucs, exp_macros, f1scores = [], [], [], [], [], [], []
    for _ in range(1):
        # seed_everything(random.randint(1, 1000000))  # use random seed for each run
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in skf.split(dataset, y):
            train_index = train_index.astype(np.int64)
            test_index = test_index.astype(np.int64)


            model_net = model_net.to(device)

            num_classes = 1024
    
            # num_features = model_net.fc.in_features #512
            num_features = model_net.classifier.in_features
            model_net.classifier = nn.Linear(num_features, num_classes)
        
            projection = Projection(262, 3)
            model = Model_SKIN(projection, model_net, 3).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_set, test_set = dataset[train_index], dataset[test_index]

            train_loader = torch.utils.data.DataLoader(train_set['data'], batch_size=400, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_set['data'], batch_size=100, shuffle=False)
            # import pdb;pdb.set_trace()

            test_micro, test_auc, test_macro = train_and_evaluate(model, train_loader, test_loader, 
                                                                  optimizer, device, loss_select)

            # logging.info(f'(Initial Performance Last Epoch) | test_micro={(test_micro * 100):.2f}, '
            #              f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}')

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--img_data_dir', type=str)
    # parser.add_argument('--skin_type', type=str)
    #parser.add_argument('--meta_data_dir', type=str, default='/home/feng/jeding/PD_contrastive_research_0817/meta_ok/')
    parser.add_argument('--model_select', type=str)
    parser.add_argument('--losses_select', type=str)
    # parser.add_argument('--dataset_choice', type=str)
    # parser.add_argument('--category', type=str)
    # parser.add_argument('--n_epoch', type=int)
    # parser.add_argument('--n_classes', type=int)
    
    args = parser.parse_args()
    train_eval(args.model_select, args.losses_select)


if __name__ == '__main__':
    main()

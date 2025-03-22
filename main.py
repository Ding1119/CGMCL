import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
from collections import Counter
from itertools import cycle
from utils import label_return
from model_resnet_skin import Projection, Model_SKIN, CNN
from losses import WeightedCrossEntropyLoss, contrastive_loss
from data_handlder.load_dataset import dataloader
import argparse

# os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

def train_eval(datadir,skin_type, loss_select, model_select , dataset_choice ,category, epoch, n_classes, exp_mode, beta, margin):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    torch.cuda.is_available()
    
 
    if model_select == 'resnet_18':
        model_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # resnet = models.resnet18(pretrained=True)
    elif model_select == 'resnet_34':
        #  model_net = models.resnet34(pretrained=True)
        model_net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_select == 'resnet_50':
        model_net = models.resnet50(pretrained=True)
    elif model_select == 'densenet':
         model_net = models.densenet121(pretrained=True)
    
    model_net = model_net.to(device)

    num_classes = 1024
    
    num_features = model_net.fc.in_features 
 
    model_net.fc = nn.Linear(num_features, num_classes) 
    model_net.fc = model_net.fc.to(device) 


    image_data_train, feature_data_train, adj_train_img, adj_f_knn_train, image_data_test, test_feature_data, adj_test_img, adj_f_knn_test = dataloader(datadir,skin_type, exp_mode)
    

    
    projection = Projection(262, 3)
    if datadir == 'skin':
        model = Model_SKIN(projection, model_net, n_classes).to(device)
    elif datadir == 'pd':
        model = Model_PD(projection, model_net, n_classes).to(device)

    class_weights = torch.full((1,n_classes),0.5).view(-1)
    criterion1 = WeightedCrossEntropyLoss(weight=class_weights)

    if loss_select == 'Contrastive_loss':
         criterion2 = contrastive_loss



    optimizer = optim.Adam(model.parameters(), lr=0.001)
    class_name = category
    n_epochs = epoch
    
    training_range = tqdm(range(n_epochs))
    

    for epoch in training_range:

        optimizer.zero_grad()

        
        image_data_train = image_data_train.to(device)
        feature_data_train = feature_data_train.to(device)
        adj_train_img = adj_train_img.to(device)
        adj_f_knn_train = adj_f_knn_train.to(device)
        
        output1, output2, emb = model(image_data_train , feature_data_train,adj_train_img, adj_f_knn_train, epoch)

   
        y = torch.tensor(label_return(dataset_choice, class_name, "train", exp_mode)).to(device)

        loss_ce1 = criterion1(output1, y)
        loss_ce2 = criterion1(output2, y)
        alpha = beta

        if loss_select == 'Contrastive_loss':
            adj = adj_train_img +  adj_f_knn_train
            diag = torch.diag(adj.sum(dim=1))
            loss_extra = criterion2( emb, adj_train_img, adj_f_knn_train, y, output1, output2, diag, n_classes, margin).to(device)
            loss = (1-alpha)*(loss_ce1 + loss_ce2) + alpha* loss_extra
            # loss = loss_ce1 + loss_ce2 + loss_extra
            # loss = loss_ce1 + loss_ce2


        loss.backward()
        optimizer.step()


    model.eval()
    with torch.no_grad():
   
        image_data_test = image_data_test.to(device)
        test_feature_data = test_feature_data.to(device)
        adj_test_img = adj_test_img.to(device)
        adj_f_knn_test = adj_f_knn_test.to(device)
        
        test_output1, test_output2, emb  = model(image_data_test, test_feature_data , adj_test_img, adj_f_knn_test,epoch)
        
        m = nn.Softmax(dim=1)
  
        test_output = test_output1 + test_output2

        pred =  m(test_output).argmax(dim=1)

        y_test = torch.from_numpy(label_return(dataset_choice ,class_name, "test", exp_mode)).to(device)
      

        correct = (pred  == y_test).sum().item()
        accuracy = correct / len(y_test)
        
 
        print(f"++++Use {model_select} model+++")

        print(classification_report(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_data_dir', type=str)
    parser.add_argument('--skin_type', type=str)
    parser.add_argument('--model_select', type=str)
    parser.add_argument('--losses_choice', type=str)
    parser.add_argument('--dataset_choice', type=str)
    parser.add_argument('--category', type=str)
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--exp_mode', type=str)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--margin', type=float)
    
    args = parser.parse_args()
    train_eval(args.img_data_dir, args.skin_type, args.losses_choice, args.model_select, args.dataset_choice,args.category, args.n_epoch, args.n_classes, args.exp_mode, args.beta, args.margin)


if __name__ == '__main__':
    main()

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
from utils import build_knn_graph, label_return, plot_auc, print_auc, calculate_metrics_new
from model import Projection, Model, CNN
from losses import WeightedCrossEntropyLoss, contrastive_loss, info_loss, MGECLoss, SACLoss
import argparse
from data_handlder.load_dataset import dataloader
# device = torch.device("mps" if torch.cuda.is_available() else "cpu")

def train_eval(datadir,skin_type, metadir, loss_select ,classes, epoch, n_classes):
    device = torch.device("mps")

    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet = resnet.to(device)
    # 将最后一层的输出维度修改为类别数目
    num_classes = 1024
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)
    resnet.fc = resnet.fc.to(device)

    image_data_train, feature_data_train, adj_train_img, adj_f_knn_train, image_data_test, test_feature_data, adj_test_img, adj_f_knn_test = dataloader(datadir,skin_type, metadir)

    projection = Projection(262, 3)
    model = Model(projection, resnet, n_classes).to(device)
    
    
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
    class_name = classes
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
        
        
        output1, output2, emb = model(image_data_train , feature_data_train,adj_train_img, adj_f_knn_train)

        y = torch.tensor(label_return(class_name, "train")).to(device)

        loss_ce1 = criterion1(output1, y)
        loss_ce2 = criterion1(output2, y)

        if loss_select == 'Contrastive_loss':
            loss_extra = criterion2( emb, adj_train_img, adj_f_knn_train, y)

        elif loss_select == 'MGEC_loss':
            adj = adj_train_img +  adj_f_knn_train
            diag = torch.diag(adj.sum(dim=1))
            loss_extra = criterion2(output1, output2, adj, diag )

        elif loss_select == 'InfoNCE_loss':
            loss_extra = criterion2( emb, adj_train_img, adj_f_knn_train, y)

        elif loss_select == 'SAC_loss':
                adj = adj_train_img +  adj_f_knn_train
                diag = torch.diag(adj.sum(dim=1))
                loss_extra = criterion2(output1, output2, adj)

        alpha = 0.4
       
        loss = (1-alpha)*(loss_ce1 + loss_ce2) + alpha* loss_extra

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

        pred =  m(test_output).argmax(dim=1)
        # test_output = test_output.argmax(dim=1)
        
        
        # y_test = torch.empty(100).random_(2)
    #     y_test = torch.tensor(label_3_test).to(device)
        y_test = torch.from_numpy(label_return(class_name, "test")).to(device)

        
        correct = (pred  == y_test).sum().item()
        accuracy = correct / len(y_test)
        
        # import pdb;pdb.set_trace()
        print(calculate_metrics_new(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
        print("Loss:", loss_select, "class_name",class_name,"Accuracy:", accuracy)
        print(classification_report(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
        
        # plot_ROC(pred.cpu().detach().numpy() , y_test.cpu().detach().numpy(), 3, classes, skin_type, loss_select)
        print_auc(pred.cpu().detach().numpy() , y_test.cpu().detach().numpy(), 3, classes, skin_type, loss_select)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_data_dir', type=str, default='/Users/test/Documents/Contrastive_PD/skin_dataset_ok/')
    parser.add_argument('--skin_type', type=str)
    parser.add_argument('--meta_data_dir', type=str, default='/Users/test/Documents/Contrastive_PD/skin_dataset_ok/meta_ok/')
    parser.add_argument('--losses_choice', type=str)
    parser.add_argument('--classes', type=str)
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--n_classes', type=int)
    
    args = parser.parse_args()
    train_eval(args.img_data_dir, args.skin_type, args.meta_data_dir, args.losses_choice, args.classes, args.n_epoch, args.n_classes)


if __name__ == '__main__':
    main()
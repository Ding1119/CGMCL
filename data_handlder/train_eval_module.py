import numpy as np
import nni
import torch
import torch.nn.functional as F
from sklearn import metrics
from typing import Optional
from torch.utils.data import DataLoader
import logging
from losses import *
from tqdm import tqdm
from data_handlder.new_dataloader import build_adj




def train_and_evaluate(model, train_loader, test_loader,
                        train_loader_f, test_loader_f,
                        train_y, test_y,
                        optimizer, device, loss_select):
    model.train()
    accs, aucs, macros = [], [], []
    epoch_num = 5
    
    for i in tqdm(range(epoch_num)):
        loss_all = 0

        adj_train = build_adj(train_loader, 300)
        adj_train_f =  build_adj(train_loader_f, 300)
        
        optimizer.zero_grad()
        # import pdb;pdb.set_trace()
        train_loader=  torch.tensor(train_loader ).float().transpose(1,3).to(device)
        train_loader_f = train_loader_f.float().to(device)
        adj_train = adj_train.to(device)
        adj_train_f = adj_train_f.to(device)
        train_y = train_y.to(device)

        output1, output2, emb = model(train_loader, train_loader_f, adj_train, adj_train_f)

        n_classes = 3
        class_weights = torch.full((1, n_classes),0.5).view(-1)
        criterion1 = WeightedCrossEntropyLoss(weight=class_weights)
        
        if loss_select == 'Contrastive_loss':
         criterion2 = contrastive_loss

        elif loss_select == 'MGEC_loss':
            criterion2 = MGECLoss()
            
        elif loss_select == 'InfoNCE_loss':
            criterion2 = info_loss
            
        elif loss_select == 'SAC_loss':
            criterion2 = SACLoss()

        optimizer.zero_grad()
        loss_ce1 = criterion1(output1, train_y)
        loss_ce2 = criterion1(output2, train_y)
        alpha = 0.4

        if loss_select == 'Contrastive_loss':
            adj = adj_train +  adj_train_f
            diag = torch.diag(adj.sum(dim=1))
            loss_extra = criterion2( emb, adj_train , adj_train_f, train_y, output1, output2, diag).to(device)
            loss = (1-alpha)*(loss_ce1 + loss_ce2) + alpha* loss_extra

        elif loss_select == 'MGEC_loss':
            adj = adj_train +  adj_train_f
            diag = torch.diag(adj.sum(dim=1))
            loss_extra = criterion2(output1, output2, adj, diag )
            loss = (1-alpha)*(loss_ce1+loss_ce2) + alpha* loss_extra
            #loss = loss_extra

        elif loss_select == 'InfoNCE_loss':
            loss_extra = criterion2( emb, adj_train ,  adj_train_f, train_y)
            loss = (1-alpha)*(loss_ce1+loss_ce2) + alpha* loss_extra

        elif loss_select == 'SAC_loss':    
            adj = adj_train +  adj_train_f
            diag = torch.diag(adj.sum(dim=1))
            loss_extra = criterion2(emb, adj)
            loss = (1-alpha)*(loss_ce1+loss_ce2) + alpha* loss_extra
        elif loss_select == 'only_CE':
            loss = loss_ce1 + loss_ce2


        loss.backward()
        optimizer.step()

        train_micro, train_auc, train_macro = evaluate(model, device, train_loader)

        # test_interval = 5
        # if (i + 1) % test_interval == 0:
        #     test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
        #     accs.append(test_micro)
        #     aucs.append(test_auc)
        #     macros.append(test_macro)
        #     text = f'(Train Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
        #            f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'

        import pdb;pdb.set_trace()

@torch.no_grad()
def evaluate(model, device, loader, test_loader: Optional[DataLoader] = None) -> (float, float):
    model.eval()
    preds, trues, preds_prob = [], [], []

    correct, auc = 0, 0
    for data in loader:
        data = data.to(device)
        c = model(data)

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
        trues += data.y.detach().cpu().tolist()
    train_auc = metrics.roc_auc_score(trues, preds_prob)

    if np.isnan(auc):
        train_auc = 0.5
    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])

    if test_loader is not None:
        test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
        return train_micro, train_auc, train_macro, test_micro, test_auc, test_macro
    else:
        return train_micro, train_auc, train_macro

def evaluatewithprint(model, device, loader, test_loader: Optional[DataLoader] = None) -> (float, float):
    model.eval()

    preds, trues, preds_prob = [], [], []

    correct, auc = 0, 0
    for data in loader:
        data = data.to(device)
        c = model(data)

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
        trues += data.y.detach().cpu().tolist()
    train_auc = metrics.roc_auc_score(trues, preds_prob)
    print(preds_prob)
    print(trues)

    if np.isnan(auc):
        train_auc = 0.5
    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])
    train_f1 = metrics.f1_score(trues, preds)
    if test_loader is not None:
        test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
        return train_micro, train_auc, train_macro, test_micro, test_auc, test_macro
    else:
        return train_micro, train_auc, train_macro, train_f1
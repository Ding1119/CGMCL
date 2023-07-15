import torch
import argparse
import os
from utils.utils import *
from utils.recorder import Recoder
from breast import BreastDataset
from load_data import dataloader_breast
import torchvision.models as models
import torch.nn as nn
from models.breast_cnn_gcn import *
from utils.losses import *
import torch.optim as optim
from tqdm import tqdm
from utils.metric import *
from sklearn.metrics import classification_report


device = torch.device("mps")
def parser_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--train_json_path", required=True)
    parser.add_argument("--val_json_path", required=True)
    parser.add_argument("--test_json_path", required=True)
    parser.add_argument("--data_dir_path", default="./dataset")
    parser.add_argument("--clinical_data_path")
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--num_classes", type=int, choices=[2, 3], default=2)

    # other
    parser.add_argument("--epoch", type=int, default=3)
    # parser.add_argument("--train_stop_auc", type=float, default=0.98)
    # parser.add_argument("--merge_method", choices=["max", "mean", "not_use"], default="mean")
    # parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--num_workers", type=int, default=8)
    
    args = parser.parse_args()

    return args


def init_dataloader(args):


    train_dataset = BreastDataset(args.train_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=30, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    
    # val_dataset = BreastDataset(args.val_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    test_dataset = BreastDataset(args.test_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    # label_test = []
    # for i in range(len(test_dataset)):
    #     label_test.append(test_dataset.__getitem__(i)["label"])
    # import pdb;pdb.set_trace()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=30, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # return train_loader, val_loader, test_loader
    return train_loader, test_loader

# def train_eval_breast(loss_select ,classes, epoch, n_classes):
#     device = torch.device("mps")
#     resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#     resnet = resnet.to(device)
#     num_classes = 1024
#     num_features = resnet.fc.in_features
#     resnet.fc = nn.Linear(num_features, num_classes)
#     resnet.fc = resnet.fc.to(device)




if __name__ == "__main__":
    args = parser_args()
    
    # train_loader, val_loader, test_loader = init_dataloader(args)
    train_loader, test_loader = init_dataloader(args)
    

    if args.num_classes > 2:
        print(f"multiple classification")
        main_fun = train_val_test_multi_class
    else:
        print(f"binary classification")
        main_fun = train_val_test_binary_class

    device = torch.device("mps")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet = resnet.to(device)
    num_classes = 1024
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)
    resnet.fc = resnet.fc.to(device)

    # training
    for epoch in range(1, args.epoch + 1):
        # scheduler.step()
        # main_fun("train", epoch, train_loader, None)
        data_loader_train = train_loader
        task_type = "train"
        n_classes = 2
        loss_select = 'Contrastive_loss'
        n_epoch = 10

        
        if task_type == "train":
            for index, item in enumerate(data_loader_train, start=1):
                bag_tensor, label = item["bag_tensor"], item["label"]
                clinical_data = item["clinical_data"][0] if "clinical_data" in item else None
                image_data_train, feature_data_train, adj_train_img, adj_f_knn_train = dataloader_breast(task_type, bag_tensor, clinical_data)
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

                optimizer = optim.Adam(model.parameters(), lr=0.001)
                n_epochs = epoch
                training_range = tqdm(range(n_epochs))

                for n_epoch in training_range:
                    optimizer.zero_grad()
                    # cnn_z  =  cnn_encoder(image_data)
                    # 前向传播
                    
                    image_data_train = image_data_train.to(device)
                    feature_data_train = feature_data_train.to(device)
                    adj_train_img = adj_train_img.to(device)
                    adj_f_knn_train = adj_f_knn_train.to(device)
                    label = label.to(device)
                    
                    
                    output1, output2, emb = model(image_data_train , feature_data_train, adj_train_img, adj_f_knn_train)
                    
                    loss_ce1 = criterion1(output1, label)
                    loss_ce2 = criterion1(output2, label )
                    
                    if loss_select == 'Contrastive_loss':
                        loss_extra = criterion2( emb, adj_train_img, adj_f_knn_train, label)

                    elif loss_select == 'MGEC_loss':
                        adj = adj_train_img +  adj_f_knn_train
                        diag = torch.diag(adj.sum(dim=1))
                        loss_extra = criterion2(output1, output2, adj, diag )

                    elif loss_select == 'InfoNCE_loss':
                        loss_extra = criterion2( emb, adj_train_img, adj_f_knn_train, label)

                    elif loss_select == 'SAC_loss':
                            adj = adj_train_img +  adj_f_knn_train
                            diag = torch.diag(adj.sum(dim=1))
                            loss_extra = criterion2(output1, output2, adj)

                    alpha = 0.4
                
                    loss = (1-alpha)*(loss_ce1 + loss_ce2) + alpha* loss_extra

                    
                    loss.backward()
                    optimizer.step()
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(n_epoch+1, n_epochs, loss.item()))

            model.eval()
            with torch.no_grad():
                data_loader_test = test_loader
                for index, item in enumerate(data_loader_test, start=1):
                    bag_tensor, label = item["bag_tensor"], item["label"]
                    clinical_data = item["clinical_data"][0] if "clinical_data" in item else None

                    image_data_test, test_feature_data, adj_test_img, adj_f_knn_test = dataloader_breast(task_type, bag_tensor, clinical_data)
                    image_data_test = image_data_test.to(device)
                    test_feature_data = test_feature_data.to(device)
                    adj_test_img = adj_test_img.to(device)
                    adj_f_knn_test = adj_f_knn_test.to(device)
                    label = label.to(device)

                    test_output1, test_output2, emb  = model(image_data_test, test_feature_data , adj_test_img, adj_f_knn_test )

                    m = nn.Softmax(dim=1)
                    # import pdb;pdb.set_trace()
                    test_output = test_output1 + test_output2
                #     test_output = emb

                    pred =  m(test_output).argmax(dim=1)
                    y_test = label

                    correct = (pred  == y_test).sum().item()
                    accuracy = correct / len(y_test)
                    
                    # import pdb;pdb.set_trace()
                    # print(calculate_metrics_new(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
                    import pdb;pdb.set_trace()
                    print(classification_report(y_test.cpu().detach().numpy(), pred.cpu().detach().numpy() ))
                        
                        # plot_ROC(pred.cpu().detach().numpy() , y_test.cpu().detach().numpy(), 3, classes, skin_type, loss_select)
                        # print_auc(pred.cpu().detach().numpy() , y_test.cpu().detach().numpy(), 3, classes, skin_type, loss_select)


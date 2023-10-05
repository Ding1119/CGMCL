from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import roc_curve,auc
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn import metrics
import torch
from itertools import cycle
import torch
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
# 转换聚类标签为PyTorch张量





def run_eval(y_test, pred_y):

  
    # import pdb;pdb.set_trace()
    [[TN, FP], [FN, TP]] = confusion_matrix(y_test, pred_y).astype(float)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    sensivity = recall = TP / (TP + FN)
    fscore = 2 * TP / (2 * TP + FP + FN)

    cluster_labels_true = torch.tensor(y_test)
    cluster_labels_pred = torch.tensor(pred_y)

    # 使用sklearn.metrics.cluster中的normalized_mutual_info_score计算NMI
    nmi = normalized_mutual_info_score(cluster_labels_true, cluster_labels_pred)


    # 假设有两个聚类结果的标签 true_labels 和 predicted_labels

    # 转换为PyTorch张量
    true_labels = torch.tensor(y_test)
    predicted_labels = torch.tensor(pred_y)
    import pdb;pdb.set_trace()
    # 使用sklearn.metrics中的adjusted_rand_score计算ARI
    ari = adjusted_rand_score(true_labels, predicted_labels)

    print("ARI:", ari)

    # print("NMI:", nmi)

    return [accuracy, precision, recall, fscore, sensivity, specificity, nmi, ari]

def label_return(dataset_choice, category,label):
    if dataset_choice == 'skin':
        if label == 'train':
            
            raw_train_label = pd.read_csv('/home/jding/Documents/PD_contrastive_research_0817/skin_dataset_ok/train_labels_df_413.csv') 
            
            return np.array(raw_train_label[f'{category}'])
        else:

            raw_test_label = pd.read_csv('/home/jding/Documents/PD_contrastive_research_0817/skin_dataset_ok/test_labels_df_395.csv')
            
            return np.array(raw_test_label[f'{category}'])
        
    elif dataset_choice == 'abide':
        
        if label == 'train':

            raw_train_label = np.load('/home/feng/jeding/PD_contrastive_research_0817/data_storage/y_train.npy') 
            
            return raw_train_label
        else:
            raw_test_label = np.load('/home/feng/jeding/PD_contrastive_research_0817/data_storage/y_test.npy') 
            
            return raw_test_label
        
    elif dataset_choice == 'pd':
        path_meta = '/home/jding/Documents/PD_contrastive_research_0817/spect_513_data' + '/' 
        raw_meta = pd.read_csv(path_meta  + 'label_513.csv') 
        label_630_id = raw_meta[raw_meta['ID'] < 634]

        label_3 = label_630_id['Lebel_3'].values
        label_2 = label_630_id['Label_2'].values

        if label == 'train':

            raw_train_label = label_3[0:300]
            
            return raw_train_label
        else:
            raw_test_label = label_3[300:]
            
            return raw_test_label


def build_knn_graph(input_data, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(input_data)
    _, indices = knn.kneighbors(input_data)
    adjacency_matrix = torch.zeros(input_data.shape[0], input_data.shape[0])
    for i, neighbors in enumerate(indices):
        adjacency_matrix[i, neighbors] = 1
        
    return adjacency_matrix


def calculate_metrics_new(gt, pred):
    """
    :param gt: 数据的真实标签，一般对应二分类的整数形式，例如:y=[1,0,1,0,1]
    :param pred: 输入数据的预测值，因为计算混淆矩阵的时候，内容必须是整数，所以对于float的值，应该先调整为整数
    :return: 返回相应的评估指标的值
    """
    """
        confusion_matrix(y_true,y_pred,labels,sample_weight,normalize)
        y_true:真实标签；
        y_pred:预测概率转化为标签；
        labels:用于标签重新排序或选择标签子集；
        sample_weight:样本权重；
        normalize:在真实（行）、预测（列）条件或所有总体上标准化混淆矩阵；
    """
    print("starting!!!-----------------------------------------------")
#     sns.set()
#     fig, (ax1, ax2) = plt.subplots(figsize=(10, 8), nrows=2)
    confusion = confusion_matrix(gt, pred)
    # 打印具体的混淆矩阵的每个部分的值
#     print(confusion)
    # 从左到右依次表示TN、FP、FN、TP
#     print(confusion.ravel())
    # 绘制混淆矩阵的图
    #sns.heatmap(confusion,annot=True)
#     ax2.set_title('sns_heatmap_confusion_matrix')
#     ax2.set_xlabel('y_pred')
#     ax2.set_ylabel('y_true')
    # fig.savefig('sns_heatmap_confusion_matrix.jpg', bbox_inches='tight')
    # 混淆矩阵的每个值的表示
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    # 通过混淆矩阵计算每个评估指标的值
    # print('AUC:',roc_auc_score(gt, pred_prob))
    print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
    print('Sensitivity:', TP / float(TP + FN))
    Sensitivity = TP / float(TP + FN)
    print('Specificity:', TN / float(TN + FP))
    Specificity = TN / float(TN + FP)
    # print('PPV:',TP / float(TP + FP))
    # PPV = TP / float(TP + FP)
    # print('NPV:',TN/float(TN + FN))
    # NPV = TN/float(TN + FN)
    # print('Recall:',TP / float(TP + FN))
    print('Precision:',TP / float(TP + FP))
    # # 用于计算F1-score = 2*recall*precision/recall+precision,这个情况是比较多的
    # P = TP / float(TP + FP)
    # R = TP / float(TP + FN)
    # print('F1-score:',(2*P*R)/(P+R))
    # print('True Positive Rate:',round(TP / float(TP + FN)))
    # print('False Positive Rate:',FP / float(FP + TN))
    print('Ending!!!------------------------------------------------------')
 
    # 采用sklearn提供的函数验证,用于对比混淆矩阵方法与这个方法的区别
    # print("the result of sklearn package")
    # AUC = roc_auc_score(gt, pred_prob)
    # print("sklearn auc:",AUC)
    # accuracy = accuracy_score(gt,pred)
    # print("sklearn accuracy:",accuracy)
    # recal = recall_score(gt,pred)
    # precision = precision_score(gt,pred)
    # print("sklearn recall:{},precision:{}".format(recal,precision))
    # print("sklearn F1-score:{}".format((2*recal*precision)/(recal+precision)))
    
    # print('---------- AUC----', AUC)
    
    return Sensitivity, Specificity

def print_auc(model_pred, y_test, n_classes ,name, image_type, loss_select):
        
#     iris = datasets.load_iris()
#     x = iris.data[:, 2:]
#     y = iris.target
#     x_test = x
#     y_test = y
#     n_classes=3
    '''NAIVE BAYES'''

    
#     input_model=model
#     input_model.fit(x_train, y_train)
#     nb=model.score(x_train, y_train)

    pred1=model_pred
    t1=sum(x==0 for x in pred1-y_test)/len(pred1)



    ### MACRO
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(pred1))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print("=========== AUC:", roc_auc["macro"])


@torch.no_grad()
def evaluate(model, device,pred, y_test) -> (float, float):
    model.eval()
    preds, trues, preds_prob = [], [], []

    # correct, auc = 0, 0
    # for data in loader:
    #     data = data.to(device)
    #     c = model(data)

    #     pred = c.max(dim=1)[1]
    #     preds += pred.detach().cpu().tolist()
    #     preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
    #     trues += data.y.detach().cpu().tolist()
    y_test = y_test.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    train_auc = metrics.roc_auc_score(y_test, pred)

    if np.isnan(auc):
        train_auc = 0.5
    train_micro = metrics.f1_score(y_test, pred, average='micro')
    train_macro = metrics.f1_score(y_test, pred, average='macro', labels=[0, 1, 3])

    # if test_loader is not None:
    #     test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
    #     return train_micro, train_auc, train_macro, test_micro, test_auc, test_macro
    # else:
    return train_micro, train_auc, train_macro



def plot_auc(model_pred, y_test, n_classes ,name, image_type, loss_select):
    
#     iris = datasets.load_iris()
#     x = iris.data[:, 2:]
#     y = iris.target
#     x_test = x
#     y_test = y
#     n_classes=3
    '''NAIVE BAYES'''

    
#     input_model=model
#     input_model.fit(x_train, y_train)
#     nb=model.score(x_train, y_train)

    pred1=model_pred
    t1=sum(x==0 for x in pred1-y_test)/len(pred1)



    ### MACRO
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(pred1))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print("=========== AUC:", roc_auc["macro"])

    lw=2
    plt.figure(figsize=(8,5))
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess',(.5,.48),color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic {image_type} for {name} via {loss_select}')
    plt.legend(loc="lower right")
    

    # return plt.show()
    return plt.savefig(f'{image_type}_{name}_{loss_select}_ROC.png')

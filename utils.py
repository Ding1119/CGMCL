from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import roc_curve,auc
# from scipy import interp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
import torch
from collections import Counter
from itertools import cycle
import torch
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_curve, auc

def label_return(dataset_choice, category,label, exp_mode):
    if dataset_choice == 'skin':
        if label == 'train':
            
            raw_train_label = pd.read_csv('dataset_skin/skin_labels/train_labels_df_413.csv') 
            # import pdb;pdb.set_trace()
            return np.array(raw_train_label[f'{category}'])
        else:

            raw_test_label = pd.read_csv('dataset_skin/skin_labels/test_labels_df_395.csv')
            # import pdb;pdb.set_trace
            return np.array(raw_test_label[f'{category}'])





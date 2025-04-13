import pandas as pd
import numpy as np
from collections import Counter

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





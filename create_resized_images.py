import pandas as pd
import numpy as np
import random 
from sklearn.linear_model import LogisticRegression
import torch 
import os 
from PIL import Image
import PIL
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch import nn

from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import image

def subset_images(data_df, K, pos_split, neg_split):
    #random.seed(42)
    data_df_1 = data_df[data_df['Cardiomegaly'] == 1]
    data_df_0 = data_df[data_df['Cardiomegaly'] == 0]
    n1 = int(pos_split * K)  # 90% of K
    n0 = int(neg_split * K)  # 10% of K

    data_df = pd.concat([data_df_1.sample(n=n1, random_state=42), data_df_0.sample(n=n0, random_state=42)])
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return data_df
    
def train_test_split_stratified(data_df, train_split=0.7, val_split=0.15, test_split=0.15): #works for both chexpert and mimic 
    random.seed(42)
    if not ('subject_id' in data_df.columns): 
        data_df['subject_id'] = data_df.Path.str.split('/').str[2] #this is only for chexpert, assumes mimic has a subject_id
        
    data_keys = list(data_df.groupby('subject_id', sort=False).groups.keys())
    
    random.shuffle(data_keys)
    
    # Split keys into training, validation, and testing sets
    train_keys = data_keys[:int(train_split * len(data_keys))]
    val_keys = data_keys[int(train_split * len(data_keys)):int((train_split + val_split) * len(data_keys))]
    test_keys = data_keys[int((train_split + val_split) * len(data_keys)):]

    train_data = data_df.loc[data_df['subject_id'].isin(train_keys)]
    val_data = data_df.loc[data_df['subject_id'].isin(val_keys)]
    test_data = data_df.loc[data_df['subject_id'].isin(test_keys)]
    
    return train_data, val_data, test_data


def create_data_chexpert(new_dir, chexpert_dir, K):
    chexpert_df = pd.read_csv(os.path.join(chexpert_dir, 'CheXpert-v1.0/train.csv'))    
    ctrain_data, cval_data, ctest_data = train_test_split_stratified(chexpert_df)
    
    ctrain_data = subset_images(ctrain_data, int(0.7*K), 0.9, 0.1) #70% of K 
    ctrain_data['split'] = ['train']*ctrain_data.shape[0]
    cval_data = subset_images(cval_data, int(0.15*K), 0.9, 0.1) #15% of K 
    cval_data['split'] = ['val']*cval_data.shape[0]
    ctest_data = subset_images(ctest_data, int(0.15*K), 0.1, 0.9) #15% of K 
    ctest_data['split'] = ['test']*ctest_data.shape[0]
    
    full_data = pd.concat([ctrain_data, cval_data, ctest_data])
    print(full_data.shape)
    for i,row in full_data.iterrows():
        img_path = os.path.join(chexpert_dir,row['Path'])
        path_name = "".join(row['Path'].split('/'))[:-4]
        img = Image.open(img_path)
        img_small = np.array(img.resize((256,256), resample=PIL.Image.BILINEAR))
        dest_path = os.path.join(new_dir, "imgs/"+path_name+".npy")
        np.save(dest_path, img_small)
        row['new_path'] = dest_path
    
    full_data.to_csv(os.path.join(new_dir,'full_data_chexpert.csv'))
    
def create_data_mimic(new_dir, mimic_info, K):
    mimic_df = pd.read_csv(os.path.join(mimic_info[0], f'mimic-cxr-jpg-2.0.0.physionet.org/paths-incl-mimic-cxr-2.0.0-{mimic_info[1]}.csv'))    
    
    mtrain_data, mval_data, mtest_data = train_test_split_stratified(mimic_df)

    mtrain_data = subset_images(mtrain_data, int(0.7*K), 0.1, 0.9) #70% of K 
    mtrain_data['split'] = ['train']*mtrain_data.shape[0]
    
    mval_data = subset_images(mval_data, int(0.15*K), 0.1, 0.9) #15% of K 
    mval_data['split'] = ['val']*mval_data.shape[0]
    
    mtest_data = subset_images(mtest_data, int(0.15*K), 0.9, 0.1) #15% of K 
    mtest_data['split'] = ['test']*mtest_data.shape[0]
    
    full_data = pd.concat([mtrain_data, mval_data, mtest_data]) #shape - K 
    print(full_data.shape)
    for i,row in full_data.iterrows():
        img_path = os.path.join(mimic_info[0],row['full_path'])
        path_name = "".join(row['full_path'].split('/'))[:-4]
        img = Image.open(img_path)
        img_small = np.array(img.resize((256,256), resample=PIL.Image.BILINEAR))
        dest_path = os.path.join(new_dir, "imgs/"+path_name+".npy")
        np.save(dest_path, img_small)
        row['new_path'] = dest_path
    
    full_data.to_csv(os.path.join(new_dir,'full_data_mimic.csv'))
    
#CSV'S ARE ALSO UPDATED TO BE MORE CONSISTENT

# chex = pd.read_csv('resized_data/chexpert/full_data_chexpert.csv', index_col=False)
# mimic = pd.read_csv('resized_data/mimic/full_data_mimic.csv', index_col=False)
# chex['original_group'] = 1 
# mimic['original_group'] = 0 
# chex = chex.rename(columns={'Path':'old_path'})
# mimic = mimic.rename(columns={'full_path':'old_path'})
# keep_cols = ['subject_id', 'Cardiomegaly', 'old_path', 'new_path', 'split', 'original_group']
# mimic[keep_cols].to_csv('resized_data/mimic/full_data_mimic.csv', index=False)
# chex[keep_cols].to_csv('resized_data/chexpert/full_data_chexpert.csv', index=False)

if __name__=="__main__":
    chexpert_dir = '/scratch/kj1447/MLHC/chexpert/chexpertchestxrays-u20210408'
    mimic_dir = '/scratch/paa9751/mlhc-project'
    label_type = 'negbio' #this can be negbio or chexpert 
    create_data_mimic('/scratch/paa9751/mlhc-project/resized_data/mimic', (mimic_dir, label_type), 10000)
    print("mimic data created")
    create_data_chexpert('/scratch/paa9751/mlhc-project/resized_data/chexpert', chexpert_dir, 10000)
    print("chexpert data created")
    
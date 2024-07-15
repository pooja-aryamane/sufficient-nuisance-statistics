import pandas as pd
import argparse
import numpy as np
import random 
#from sklearn.linear_model import LogisticRegression
import torch 
import os 
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch import nn

from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import image
import logging

from dataloader import * 
from models import * 


class CombinedDataset(Dataset):
    def __init__(self, combined_df, border_sz):
        self.border_size = border_sz
        self.image_paths = combined_df.new_path.tolist()
        self.image_labels = combined_df.Cardiomegaly.tolist()
        self.basic_transformation = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image_data = np.load(image_path)
                    
        image_data = self.basic_transformation(image_data)
        
        image_data[:, self.border_size:-self.border_size, self.border_size:-self.border_size] = 0
        
        return image_data, self.image_labels[index]
    
def get_dataloader(combined_df, border_sz):
    dataset = CombinedDataset(combined_df, border_sz)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader 

def get_predictions(maxacc_model, minloss_model, border_sz, split_idx,device):
    chexpert_df = pd.read_csv('/scratch/paa9751/mlhc-project/resized_data/chexpert/full_data_chexpert_full.csv')
    mimic_df = pd.read_csv('/scratch/paa9751/mlhc-project/resized_data/mimic/full_data_mimic_full.csv')
    sigmoid = torch.nn.Sigmoid()
    model_max_acc = torch.load(maxacc_model, map_location=device)
    model_min_loss = torch.load(minloss_model, map_location=device)
    model_max_acc.to(device)
    model_max_acc.eval()
    model_min_loss.to(device)
    model_min_loss.eval()
    
    preds_maxacc=[]
    preds_minloss=[]
    combined = pd.concat([chexpert_df,mimic_df])
    combined = combined[combined.sample_split==split_idx]
    #combined = combined[combined.split=="train"] #only train set 
    dataloader = get_dataloader(combined, border_sz)
    with torch.no_grad(): 
        for idx, (batchx, batchy) in enumerate(dataloader): 
            pred_maxacc = model_max_acc(batchx.to(device))
            pred_minloss = model_min_loss(batchx.to(device))
            preds_maxacc.append(pred_maxacc.item())
            preds_minloss.append(pred_minloss.item())
            if idx%100==0:
                print(f'{idx*100/len(dataloader)}% pred done')
    combined['max_accuracy_model_predictions'] = preds_maxacc
    combined['min_loss_model_predictions'] = preds_minloss
    return combined 

def get_full_data_predictions(modelpath,device): #get preds on train,val,test
    chexpert_df = pd.read_csv('/scratch/paa9751/mlhc-project/resized_data/chexpert/full_data_chexpert_full.csv')
    mimic_df = pd.read_csv('/scratch/paa9751/mlhc-project/resized_data/mimic/full_data_mimic_full.csv')
    sigmoid = torch.nn.Sigmoid()
    model = torch.load(modelpath, map_location=device)
    model.to(device)
    model.eval()
    preds=[]
    combined = pd.concat([chexpert_df,mimic_df])
    dataloader = get_dataloader(combined,0)
    with torch.no_grad(): 
        for idx, (batchx, batchy) in enumerate(dataloader): 
            pred = model(batchx.to(device))
            preds.append(pred.item())
            if idx%100==0:
                print(f'{idx*100/len(dataloader)}% pred done')
    combined['predictions'] = preds
    return combined 

def get_pred_df_both(model_paths, border_sz, device): 
    chexpert_df = pd.read_csv('resized_data/chexpert/full_data_chexpert.csv')
    mimic_df = pd.read_csv('resized_data/mimic/full_data_mimic.csv')
    sigmoid = torch.nn.Sigmoid()

    model_sns = torch.load(model_paths[0], map_location=device)
    model_gdro = torch.load(model_paths[1], map_location=device)
    
    model_sns.to(device)
    model_sns.eval()

    model_gdro.to(device)
    model_gdro.eval()

    preds_sns=[]
    preds_gdro=[]
    combined = pd.concat([chexpert_df,mimic_df])
    #combined = combined[combined.split=="train"] #only train set 
    dataloader = get_dataloader(combined, border_sz)
    with torch.no_grad(): 
        for idx, (batchx, batchy) in enumerate(dataloader): 
            predsns = model_sns(batchx.to(device))
            predsns = sigmoid(predsns)
            preds_sns.append(predsns.item())
            
            pred_gdro = model_gdro(batchx.to(device))
            pred_gdro = sigmoid(pred_gdro)
            preds_gdro.append(pred_gdro.item())
            
            if idx%100==0:
                print(f'{idx*100/len(dataloader)}% pred done')
    combined['predicted_prob_sns'] = preds_sns
    combined['predicted_prob_true'] = preds_gdro
    
    return combined
    
def create_sns_group_df(combined_df, thresh): 
    
    combined['sns_group'] = [0 if p<thresh else 1 for p in combined_df.predicted_prob]
    
    print('saving dataframe...')
    combined.to_csv('resized_data/combined_df_sns_groups.csv')
    
    return combined 

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelpath1='balanced-erm-new/new_weightdecay_0.001+pretrained_True+batchsize_256+lr_0.0001+transform_True_max_val_acc.pth'
    modelpath2='balanced-erm-new/new_weightdecay_0.01+pretrained_True+batchsize_256+lr_0.0001+transform_True_max_val_acc.pth'

    print('getting preds..')
    df_pred1 = get_full_data_predictions(modelpath1,device)
    df_pred2 = get_full_data_predictions(modelpath2,device)
    print('pred done, saving combined df')
    df_pred1.to_csv('balanced-erm-new/lowwd_balanced_predictions.csv', index=False)
    df_pred2.to_csv('balanced-erm-new/higherwd_balanced_predictions.csv', index=False)

    
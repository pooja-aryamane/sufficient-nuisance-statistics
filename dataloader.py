import pandas as pd
import numpy as np
import random 
from sklearn.linear_model import LogisticRegression
import torch 
import os 
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset,WeightedRandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch import nn

from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import image

class GetDataset(Dataset):
    
    def __init__(self, paths_col, dataframe, image_size, normalization, transform, border_size, group_cname):
        """
        Init Dataset
        
        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """
        
        dataframe = dataframe[(dataframe.Cardiomegaly==1)|(dataframe.Cardiomegaly==0)]
        
        # if split_idx!=None: 
        #     dataframe = dataframe[dataframe.sample_split==split_idx]
        #     print(dataframe.shape[0])

        self.border_size = border_size
        self.transform = transform
        self.groups = []
        
        self.groups= dataframe[group_cname].tolist()
        self.true_groups = dataframe['true_group_idx'].tolist()
        
        self.image_paths = dataframe[paths_col].tolist()
        self.image_labels = dataframe.Cardiomegaly.tolist()
        
        self.basic_transformation = transforms.ToTensor()
        self.normalize = normalization
        
        if transform: 
            # image_transformation = [NoneTransform(),
            #         transforms.ElasticTransform(alpha=250.0), 
            #         transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)), 
            #         transforms.RandomInvert(), 
            #         transforms.RandomHorizontalFlip(p=0.5), 
            #         transforms.RandomVerticalFlip(p=0.5), 
            #         transforms.RandomRotation(degrees=(0, 180)), 
            #         transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))]

            # self.image_transformation = transforms.RandomChoice(image_transformation)

            image_transformation = [NoneTransform(),
                    transforms.ElasticTransform(alpha=250.0), 
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)), 
                    transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.RandomVerticalFlip(p=0.5), 
                    transforms.RandomRotation(degrees=(0, 180))]
            
            self.image_transformation = transforms.RandomChoice(image_transformation, [0.3, 0.14,0.14,0.14,0.14,0.14])
                
        if normalization:
            # Normalization with mean and std from ImageNet
            # IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
            # IMAGENET_STD = [0.229, 0.224, 0.225]
            IMAGENET_MEAN = 0.445         # Mean of ImageNet dataset (used for normalization)
            IMAGENET_STD = 0.269
            self.image_normalization = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        
        # Read image
        image_path = self.image_paths[index]
#         image_data = Image.open(image_path).convert("RGB") # Convert image to RGB channels #save these as arrays!

        image_data = np.load(image_path)

        #image_data = torch.from_numpy(image_data)
        
        #image_data = image_data.unsqueeze(0)
                    
        image_data = self.basic_transformation(image_data)
        
        if self.transform:           
            image_data = self.image_transformation(image_data)

        if self.normalize:           
            image_data = self.image_normalization(image_data)

        image_data[:, self.border_size:-self.border_size, self.border_size:-self.border_size] = 0

        return image_path, image_data, self.image_labels[index], self.groups[index], self.true_groups[index]
    

class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None
    
    Args:
        image in, image out, nothing is done
    """
    def __call__(self, image):  
        return image
        
class CheXPertDataset(Dataset):
    
    def __init__(self, paths_col, dataframe, image_size, normalization, transform, border_size, group_cname, split_idx):
        """
        Init Dataset
        
        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """
        
        dataframe = dataframe[(dataframe.Cardiomegaly==1)|(dataframe.Cardiomegaly==0)]
        
        if split_idx!=None: 
            dataframe = dataframe[dataframe.sample_split==split_idx]
            print(dataframe.shape[0])

        self.border_size = border_size
        self.transform = transform
        self.groups = []
        
        self.groups= dataframe[group_cname].tolist()
        self.true_groups = dataframe['true_group_idx'].tolist()
        
        self.image_paths = dataframe[paths_col].tolist()
        self.image_labels = dataframe.Cardiomegaly.tolist()
        
        self.basic_transformation = transforms.ToTensor()
        self.normalize = normalization
        
        if transform: 
            # image_transformation = [NoneTransform(),
            #         transforms.ElasticTransform(alpha=250.0), 
            #         transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)), 
            #         transforms.RandomInvert(), 
            #         transforms.RandomHorizontalFlip(p=0.5), 
            #         transforms.RandomVerticalFlip(p=0.5), 
            #         transforms.RandomRotation(degrees=(0, 180)), 
            #         transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))]

            # self.image_transformation = transforms.RandomChoice(image_transformation)

            image_transformation = [NoneTransform(),
                    transforms.ElasticTransform(alpha=250.0), 
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)), 
                    transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.RandomVerticalFlip(p=0.5), 
                    transforms.RandomRotation(degrees=(0, 180))]
            
            self.image_transformation = transforms.RandomChoice(image_transformation, [0.3, 0.14,0.14,0.14,0.14,0.14])
                
        if normalization:
            # Normalization with mean and std from ImageNet
            # IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
            # IMAGENET_STD = [0.229, 0.224, 0.225]
            IMAGENET_MEAN = 0.445         # Mean of ImageNet dataset (used for normalization)
            IMAGENET_STD = 0.269
            self.image_normalization = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        
        # Read image
        image_path = self.image_paths[index]
#         image_data = Image.open(image_path).convert("RGB") # Convert image to RGB channels #save these as arrays!

        image_data = np.load(image_path)

        #image_data = torch.from_numpy(image_data)
        
        #image_data = image_data.unsqueeze(0)
                    
        image_data = self.basic_transformation(image_data)
        
        if self.transform:           
            image_data = self.image_transformation(image_data)

        if self.normalize:           
            image_data = self.image_normalization(image_data)

        image_data[:, self.border_size:-self.border_size, self.border_size:-self.border_size] = 0

        return image_path, image_data, self.image_labels[index], self.groups[index], self.true_groups[index]
    
    
class MIMICDataset(Dataset):
    
    def __init__(self,paths_col, dataframe, image_size, normalization, transform, border_size, group_cname,split_idx):
        """
        Init Dataset
        
        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """
        dataframe = dataframe[(dataframe.Cardiomegaly==1)|(dataframe.Cardiomegaly==0)]
        if split_idx!=None: 
            dataframe = dataframe[dataframe.sample_split==split_idx]
            print(dataframe.shape[0])

        self.border_size = border_size
        self.transform = transform
        self.groups = []
        
        self.groups= dataframe[group_cname].tolist()
        self.true_groups = dataframe['true_group_idx'].tolist()
        
        self.image_paths = dataframe[paths_col].tolist()
        self.image_labels = dataframe.Cardiomegaly.tolist()
        
        self.basic_transformation = transforms.ToTensor()

        self.normalize = normalization
        
        if transform: 
            # image_transformation = [transforms.ElasticTransform(alpha=250.0), 
            #         transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)), 
            #         transforms.RandomInvert(), 
            #         transforms.RandomHorizontalFlip(p=0.5), 
            #         transforms.RandomVerticalFlip(p=0.5), 
            #         transforms.RandomRotation(degrees=(0, 180)), 
            #         transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))]

            # self.image_transformation = transforms.RandomChoice(image_transformation)

            image_transformation = [NoneTransform(),
                    transforms.ElasticTransform(alpha=250.0), 
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)), 
                    transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.RandomVerticalFlip(p=0.5), 
                    transforms.RandomRotation(degrees=(0, 180))]
            
            self.image_transformation = transforms.RandomChoice(image_transformation, [0.5, 0.1,0.1,0.1,0.1,0.1])
                
        if normalization:
            # Normalization with mean and std from ImageNet
            # IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
            # IMAGENET_STD = [0.229, 0.224, 0.225]
            IMAGENET_MEAN = 0.445         # Mean of ImageNet dataset (used for normalization)
            IMAGENET_STD = 0.269
            self.image_normalization = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        # Read image
        image_path = self.image_paths[index]
#         image_data = Image.open(image_path).convert("RGB") # Convert image to RGB channels #save these as arrays!

        image_data = np.load(image_path)

        #image_data = torch.from_numpy(image_data)
        
        #image_data = image_data.unsqueeze(0)
    
        image_data = self.basic_transformation(image_data) # 1 x 256 x 256 
        
        if self.transform:           
            image_data = self.image_transformation(image_data)

        if self.normalize:           
            image_data = self.image_normalization(image_data)

        image_data[:, self.border_size:-self.border_size, self.border_size:-self.border_size] = 0

        return image_path, image_data, self.image_labels[index], self.groups[index], self.true_groups[index]
    
    
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


def load_combined_data(chexpert_dir, mimic_info, IMAGE_SIZE, TRANSFORM, NWORKERS, BATCH_SIZE, BORDER_SZ, K):
    chexpert_df = pd.read_csv(os.path.join(chexpert_dir, 'CheXpert-v1.0/train.csv')) #make sure chexpert paths are from /scratch..
    mimic_df = pd.read_csv(os.path.join(mimic_info[0], f'paths-incl-mimic-cxr-2.0.0-{mimic_info[1]}.csv'))

    ctrain_data, cval_data, ctest_data = train_test_split_stratified(chexpert_df)
    mtrain_data, mval_data, mtest_data = train_test_split_stratified(mimic_df)
    
    ctrain_data = subset_images(ctrain_data, int(0.7*K), 0.9, 0.1) #70% of K 
    cval_data = subset_images(cval_data, int(0.15*K), 0.9, 0.1) #15% of K 
    ctest_data = subset_images(ctest_data, int(0.15*K), 0.1, 0.9) #15% of K 
    
    mtrain_data = subset_images(mtrain_data, int(0.7*K), 0.1, 0.9)
    mval_data = subset_images(mval_data, int(0.15*K), 0.1, 0.9)
    mtest_data = subset_images(mtest_data, int(0.15*K), 0.9, 0.1)

    mimic_train_dataset = MIMICDataset('old_path', mtrain_data, IMAGE_SIZE, TRANSFORM, border_size=BORDER_SZ)
    
    mimic_val_dataset = MIMICDataset('old_path', mval_data, IMAGE_SIZE, TRANSFORM, border_size=BORDER_SZ)
    
    mimic_test_dataset = MIMICDataset('old_path', mtest_data, IMAGE_SIZE, TRANSFORM,border_size=BORDER_SZ)

    chexpert_train_dataset = CheXPertDataset('old_path', ctrain_data, IMAGE_SIZE, TRANSFORM,
                                             border_size=BORDER_SZ) 
    
    chexpert_val_dataset = CheXPertDataset('old_path', cval_data, IMAGE_SIZE, TRANSFORM,
                                           border_size=BORDER_SZ)
    
    chexpert_test_dataset = CheXPertDataset('old_path', ctest_data, IMAGE_SIZE, TRANSFORM,
                                           border_size=BORDER_SZ)
    
    combined_train_dataset = ConcatDataset([mimic_train_dataset, chexpert_train_dataset])
    combined_val_dataset = ConcatDataset([mimic_val_dataset, chexpert_val_dataset])
    combined_test_dataset = ConcatDataset([mimic_test_dataset, chexpert_test_dataset])
  
    train_dataloader = DataLoader(dataset=combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NWORKERS, pin_memory=True)
    val_dataloader = DataLoader(dataset=combined_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NWORKERS, pin_memory=True)
    test_dataloader = DataLoader(dataset=combined_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NWORKERS, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader 


def load_resized_data(chexpert_dir, mimic_dir, IMAGE_SIZE, NORMALISE, TRANSFORM, NWORKERS, BATCH_SIZE, BORDER_SZ, GROUP_CNAME, split_idx=None, sampler=None):
    chexpert_df = pd.read_csv(chexpert_dir)
    mimic_df = pd.read_csv(mimic_dir)
    
    mimic_train_dataset = MIMICDataset('new_path', mimic_df[mimic_df.split=='train'], IMAGE_SIZE, NORMALISE, TRANSFORM, BORDER_SZ, GROUP_CNAME,split_idx)
    
    mimic_val_dataset = MIMICDataset('new_path', mimic_df[mimic_df.split=='val'], IMAGE_SIZE, NORMALISE, False, BORDER_SZ, GROUP_CNAME,None)
    
    mimic_test_dataset = MIMICDataset('new_path', mimic_df[mimic_df.split=='test'], IMAGE_SIZE, NORMALISE, False, BORDER_SZ, GROUP_CNAME,None)

    chexpert_train_dataset = CheXPertDataset('new_path', chexpert_df[chexpert_df.split=='train'], IMAGE_SIZE, NORMALISE, TRANSFORM, BORDER_SZ, GROUP_CNAME,split_idx) 
    
    chexpert_val_dataset = CheXPertDataset('new_path', chexpert_df[chexpert_df.split=='val'], IMAGE_SIZE, NORMALISE, False, BORDER_SZ, GROUP_CNAME,None)
    
    chexpert_test_dataset = CheXPertDataset('new_path', chexpert_df[chexpert_df.split=='test'], IMAGE_SIZE, NORMALISE, False, BORDER_SZ, GROUP_CNAME,None)
    
    combined_train_dataset = ConcatDataset([mimic_train_dataset, chexpert_train_dataset])
    combined_val_dataset = ConcatDataset([mimic_val_dataset, chexpert_val_dataset])
    combined_test_dataset = ConcatDataset([mimic_test_dataset, chexpert_test_dataset])

    train_dataloader = DataLoader(dataset=combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NWORKERS, pin_memory=True, sampler=sampler)
    val_dataloader = DataLoader(dataset=combined_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NWORKERS, pin_memory=True, sampler=sampler)
    test_dataloader = DataLoader(dataset=combined_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NWORKERS, pin_memory=True, sampler=sampler)
    
    return train_dataloader, val_dataloader, test_dataloader 

def get_loaders_rw(chexpert_dir, mimic_dir, IMAGE_SIZE, NORMALISE, TRANSFORM, NWORKERS, BATCH_SIZE, BORDER_SZ, GROUP_CNAME):
    chexpert_df = pd.read_csv(chexpert_dir)
    mimic_df = pd.read_csv(mimic_dir)
    df = pd.concat([chexpert_df,mimic_df])
    tr_sampler = WeightedRandomSampler(list(df[df.split=='train'].clipped_weight.values), len(list(df[df.split=='train'].clipped_weight.values)))
    va_sampler = WeightedRandomSampler(list(df[df.split=='val'].clipped_weight.values), len(list(df[df.split=='val'].clipped_weight.values)))
    te_sampler = WeightedRandomSampler(list(df[df.split=='test'].clipped_weight.values), len(list(df[df.split=='test'].clipped_weight.values)))
    
    trainds = GetDataset('new_path', df[df.split=='train'], IMAGE_SIZE, NORMALISE, TRANSFORM, BORDER_SZ, GROUP_CNAME)
    valds = GetDataset('new_path', df[df.split=='val'], IMAGE_SIZE, NORMALISE, TRANSFORM, BORDER_SZ, GROUP_CNAME)
    testds = GetDataset('new_path', df[df.split=='test'], IMAGE_SIZE, NORMALISE, TRANSFORM, BORDER_SZ, GROUP_CNAME)

    train_dataloader = DataLoader(dataset=trainds, batch_size=BATCH_SIZE,num_workers=NWORKERS, pin_memory=True, sampler=tr_sampler)
    val_dataloader = DataLoader(dataset=valds, batch_size=BATCH_SIZE, num_workers=NWORKERS, pin_memory=True, sampler=va_sampler)
    test_dataloader = DataLoader(dataset=testds, batch_size=BATCH_SIZE, num_workers=NWORKERS, pin_memory=True, sampler=te_sampler)
    
    return train_dataloader, val_dataloader, test_dataloader 
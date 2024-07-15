import pandas as pd
import numpy as np
import random 
from sklearn.linear_model import LogisticRegression
import torch 
import os 
from PIL import Image
from torchvision import models
from torch import nn

from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import image

class Resnet18(torch.nn.Module):  
    def __init__(self, num_classes, pretrained, embed_dim=512):
        super(Resnet18, self).__init__()
        if pretrained: 
            self.resnet18 = models.resnet18('IMAGENET1K_V1')
        else:
            self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18 = nn.Sequential(*(list(self.resnet18.children())[:-1]))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x): 
        #x = self.conv_map(x)
        x = self.resnet18(x) 
        x = x.squeeze(2).squeeze(2)
        out = self.fc_out(x)
        return out
    
class Resnet50(torch.nn.Module):  
    def __init__(self, num_classes, pretrained, embed_dim=2048):
        super(Resnet50, self).__init__()
        if pretrained: 
            self.resnet50 = models.resnet50('IMAGENET1K_V1')
        else:
            self.resnet50 = models.resnet50(weights=None)
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50 = nn.Sequential(*(list(self.resnet50.children())[:-1]))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x): 
        #x = self.conv_map(x)
        x = self.resnet50(x) 
        x = x.squeeze(2).squeeze(2)
        out = self.fc_out(x)
        return out
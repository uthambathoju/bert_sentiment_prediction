
import os
import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F

os.chdir('D:/Utham/kaggle/bengali_ai')


class ResNet34(nn.Module):
    def __init__(self , pretrained):
        #whenever u define torch models u inherit from nn.Module
        super(ResNet34 , self, pretrained).__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = 'imagenet')
        else:
            """
            If u use torchvision , the last layer is FC
            If u use pretrained models , u will have last linear : 
            Linear(in_features = 512 , out_features=1000, bias=True) , We need to change that
            """
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = None)
        #as we have 168 Graphemes , 11 vowels , 7 consonants
        self.l0 = nn.Linear(512 , 168)
        self.l1 = nn.Linear(512 , 11)
        self.l2 = nn.Linear(512 , 7)

    
    #forward function for above 3 lines which takes a batch
    def forward(self , X):
        #batch size , channels(not needed) , height(not needed) , width(not needed) = x.shape
        bs,  _,  _, _   = X.shape
        #batch_size = X.shape
        X = self.model.features(X)
        X = F.adaptive_avg_pool2d(input = X , output_size =1).reshape(bs , -1)
        l0 = self.l0(X)
        l1 = self.l1(X)
        l2 = self.l2(X)
        return l0 , l1 ,l2 



        
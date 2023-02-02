"""
  -This script gets all items from the training dataset
  - we use pytorch for this problem
  - we make it general if we want to change the model at some point , we can change it very easily
"""

import os
import joblib
import numpy as np
import pandas as pd
import albumentations
import torch

from PIL import Image


os.chdir('D:/Utham/kaggle/bengali_ai')

class TrainBengaliDataset:
    def __init__(self,
                 folds ,
                 img_height , 
                 img_width ,
                 mean ,
                 std
                 ):
        #for image problems , we always need to normalize(mean , std)
        df = pd.read_csv('./input/train_folds.csv')
        df =df[['image_id' ,'grapheme_root' , 'vowel_diacritic' , 'consonant_diacritic' , 'kfold']]

        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        """
            - augmentations has to be done seperately for train and validation
            - u dont want to have augmentations for validation until and unless u run it several times on average
            - if length of folds is 1 that means i'm in validation phase
        """
        if(len(folds) == 1):
            self.aug = albumentations.Compose([
                    albumentations.Resize(img_height , img_width , always_apply=True), 
                    albumentations.Normalize(mean , std , always_apply=True)
            ])
        else:
            #this block is for training dataset
            #p -> probability , when to apply this so, 90% of the times it is applying this shiftscalerotate
            self.aug = albumentations.Compose([
                    albumentations.Resize(img_height , img_width , always_apply=True), 
                    albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.1 ,
                                                rotate_limit=5,
                                                p=0.9),
                    albumentations.Normalize(mean , std , always_apply=True)
            ])

    def __len__(self):
        return len(self.image_ids)

    # item - item index i,e num (0 - length of the dataset)
    def __getitem__(self , item):
        image = joblib.load(f"./input/image_pickles/{self.image_ids[item]}.pkl")
        #image is a vector , so reshape it from 1D vector to 2D array(137 , 236) provided in kaggle data
        image = image.reshape(137 , 236).astype(float)
        """
            - convert given numpy array to PIL image(RGB - gray scale image single channel RGB),
                bec'coz most of the models that v have are pre-trained from .vision and pre-trained models
                they all work on RGB
        """
        image = Image.fromarray(image).convert("RGB")
        #this line will apply augmentation and take image as an output
        image = self.aug(image = np.array(image))['image']
        #transpose the image to fit the torchvision models , sanity check- converted into float
        #to understand more about below line , learn how torchvision models expects the images
        image = np.transpose(image , (2 , 0 , 1)).astype(np.float32)
        return {
            'image' : torch.tensor(image , dtype = torch.float),
            'grapheme_root' : torch.tensor(self.grapheme_root[item] , dtype = torch.long),
            'vowel_diacritic' : torch.tensor(self.vowel_diacritic[item] , dtype = torch.long),
            'consonant_diacritic' : torch.tensor(self.consonant_diacritic[item] , dtype = torch.long)
        }


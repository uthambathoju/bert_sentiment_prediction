import os
import sys

#os.environ['WRK_DIR'] = 'D:/Utham/ML_proj/categorical_encoding/'

#os.environ['CUDA_VISIBLE_DEVICES'] = 0,1 , if u have multiple cuda devices

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IMG_HEIGHT'] = '137'
os.environ['IMG_WIDTH'] = '236'
os.environ['EPOCHS'] = '50'
os.environ["TRAIN_BATCH_SIZE"] = '64' # if GPU can fit more train data increase it to 128 , 256
os.environ["TEST_BATCH_SIZE"] = '8'
os.environ["MODEL_MEAN"] = "(0.485 , 0.456 , 0.406)"
os.environ["MODEL_STD"] = "(0.229 , 0.224 , 0.225)"
os.environ["BASE_MODEL"] = "resnet34"
os.environ["TRAINING_FOLDS_CSV"] = "./input/train_folds.csv"


os.environ["TRAINING_FOLDS"] = "(0 ,1 , 2, 3)"
os.environ["VALIDATION_FOLDS"] = "(4,)"
#python train.py
"""
os.environ["TRAINING_FOLDS"] = "(0 ,1 , 4, 3)"
os.environ["VALIDATION_FOLDS"] = "(2,)"
#python train.py

os.environ["TRAINING_FOLDS"] = "(0 ,1 , 2, 4)"
os.environ["VALIDATION_FOLDS"] = "(3,)"
#python train.py

os.environ["TRAINING_FOLDS"] = "(0 ,4 , 2, 3)"
os.environ["VALIDATION_FOLDS"] = "(1,)"
#python train.py

os.environ["TRAINING_FOLDS"] = "(4 ,1 , 2, 3)"
os.environ["VALIDATION_FOLDS"] = "(0,)"
#python train.py
"""

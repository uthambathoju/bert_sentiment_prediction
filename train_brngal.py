"""
training and validation loops 
required params :
   - train folds csv file
   - image height
   - image width 
   - no of epochs , batch size for train and test
   - model mean , std
   - training folds and validation folds

ast.literal_eval : what is does is , if there is a string of lists or tuples
                   its going to convert it to from string lists/tuples to normal lists/tuples
"""
import os
import ast
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import TrainBengaliDataset
from model_dispatcher import MODEL_DISPATCHER

import data_config

os.chdir('D:/Utham/kaggle/bengali_ai/')

#define required params
DEVICE = 'cuda'
TRAINING_FOLDS_CSV = os.environ.get('TRAINING_FOLDS_CSV')
IMG_HEIGHT = int(os.environ.get('IMG_HEIGHT'))
IMG_WIDTH  = int(os.environ.get('IMG_WIDTH'))
EPOCHS     = int(os.environ.get('EPOCHS'))

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
TEST_BATCH_SIZE  = int(os.environ.get('TEST_BATCH_SIZE'))

MODEL_MEAN = ast.literal_eval(os.environ.get('MODEL_MEAN'))
MODEL_STD  = ast.literal_eval(os.environ.get('MODEL_STD'))

TRAINING_FOLDS   = ast.literal_eval(os.environ.get('TRAINING_FOLDS'))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get('VALIDATION_FOLDS'))
BASE_MODEL       = os.environ.get('BASE_MODEL')


# we can also do weighted avg loss that gives u better results
def loss_fn(outputs , targets):
    o1 , o2 , o3 = outputs
    t1 , t2 , t3 = targets
    l1 = nn.CrossEntropyLoss()(o1 , t1)
    l2 = nn.CrossEntropyLoss()(o2 , t2)
    l3 = nn.CrossEntropyLoss()(o3 , t3)
    #avg -loss
    return (l1 + l2 + l3) / 3


#u might want a scheduler sometimes in the loop ,if it is a different scheduler
def train(dataset , data_loader , model , optimizer):
    model.train()
    for batch_idx , dataset in tqdm(enumerate(data_loader) , total = int(len(dataset) / data_loader.batch_size)):
        image = dataset['image']
        grapheme_root = dataset['grapheme_root']
        vowel_diacritic = dataset['vowel_diacritic']
        consonant_diacritic = dataset['consonant_diacritic']

        #put these into the cuda device
        """
        image = image.to(DEVICE , dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE , dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE , dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE , dtype = torch.long)
        """
        #train the model
        """
           - make sure u r output maintained same as models.py forard functiond (return l0 , l1 ,l2)
             otherwise it  is going to give some wrong loss
        """
        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root , vowel_diacritic , consonant_diacritic)
        loss = loss_fn(outputs ,targets)
         
        loss.backward()
        optimizer.step()

 
def evaluate(dataset , data_loader , model):
    model.eval()
    final_loss = 0
    counter = 0
    """
    - we have done it in simple way , it is better if we use the actual metric of the competition and loss too
    """
    for batch_idx , dataset in tqdm(enumerate(data_loader) , total = int(len(dataset) / data_loader.batch_size)):
        counter = counter + 1
        image = dataset['image']
        grapheme_root = dataset['grapheme_root']
        vowel_diacritic = dataset['vowel_diacritic']
        consonant_diacritic = dataset['consonant_diacritic']

        #put these into the cuda device
        """
        image = image.to(DEVICE , dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE , dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE , dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE , dtype = torch.long)
        """

        #train the model
        """
          - make sure u r output maintained same as models.py forard functiond (return l0 , l1 ,l2)
                otherwise it  is going to give some wrong loss
        """
        outputs = model(image)
        targets = (grapheme_root , vowel_diacritic , consonant_diacritic)
        loss = loss_fn(outputs ,targets)
        finalloss  += loss
    return final_loss/counter


def main():
    #load the model
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True) #True be'coz it is a training loop
    #model.to(DEVICE)
    
    #DataLoader for Train/validation
    train_dataset = TrainBengaliDataset(
        folds = TRAINING_FOLDS,
        img_height = IMG_HEIGHT ,
        img_width  = IMG_WIDTH , 
        mean  = MODEL_MEAN , 
        std = MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset ,
        batch_size = TRAIN_BATCH_SIZE ,
        shuffle=True , 
        num_workers=4,
    )

    #validation
    valid_dataset = TrainBengaliDataset(
        folds = VALIDATION_FOLDS,
        img_height = IMG_HEIGHT ,
        img_width  = IMG_WIDTH , 
        mean  = MODEL_MEAN , 
        std = MODEL_STD
    )
    
    """
     - shuffle true/fasle doesn't matter for validation because the dataset object
       is returning the labels even for validation.
    """
     
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset ,
        batch_size = TEST_BATCH_SIZE ,
        shuffle=False , 
        num_workers=4,
    )

    #Optimizer
    # train on all parameters , u can experiment with specific params and set different lr for different layers
    optimizer = torch.optim.Adam(model.parameters() , lr = 1e-4) 
    #scheduler 
    """
    NOTE : when u use scheduler from torch keep in mind that some schedulers need to
    step after every batch , some after every epoch .its very important
    """
    #reducelronplateau - when its splaterring(drowning) my model scores then i reduce the learning rate 
    #mode can be min/max ,mode is min be'coz we will see recall score , based on it we'll say reduce the learning rate
    #factor - reduce it by a factor of 0.3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer ,
                                                           mode="min" , 
                                                           patience=5 ,
                                                           factor=0.3 , 
                                                           verbose=True
                                                           )
    
    #if u have multiple GPUs in the system , u can convert u r model to data parallel
    if torch.cuda.device_count() > 1:
        model =nn.DataParallel(model)

    """
     - u can implement multiple things like EarlyStopping
       https://github.com/Bjarten/early-stopping-pytorch
    """
     
    for epoch in range(EPOCHS):
        train(train_dataset , train_loader , model , optimizer)
        val_score = evaluate(valid_dataset , valid_loader , model)
        scheduler.step(val_score)
        # u can save it with .pkl/.h5 as well
        torch.save(model.state_dict() , f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}..bin")


if __name__ == '__main__':
    main()










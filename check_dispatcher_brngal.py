import os
os.chdir('D:/Utham/kaggle/bengali_ai/src')

from model_dispatcher import MODEL_DISPATCHER

model =MODEL_DISPATCHER['resnet34'](pretrained  = False)

print(model)
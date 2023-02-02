"""
    - define the model in such a way that u can reuse it and u can also experiment  a little bit more
    - take a look at bert from transformers which has good documentation
"""
import transformers  #transformers for the bert model itself
import torch.nn as nn   

import os
os.chdir('D:/Utham/kaggle/bert_sentiment_prediction/')
#user-defined 
import config

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased , self).__init__()
        #load the model
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        #already we knew that bert base model that we use has 768 output features and 1(binary classification)
        self.out = nn.Linear(768 , 1)

    #Bert Model takes different kinds of inputs
    def forward(self , ids , mask , token_type_ids):
        #u'll get two outputs 
        """
            - out1(last hidden states) = sequence of hidden states for each and every token
            - lets say if u have 512 tokens which is our max_length , u'll have 512 vectors of size 768
              and that's for each batch
            - out2 is called pooler output from the bert pooler layer
            - u can perform some kind of average pooling / max pooling 
        """
       #out1(remove this) , out2   
        _ , o2 = self.bert(
            ids = ids,
            attention_mask = mask ,
            token_type_ids = token_type_ids
            )
      #o2 will give u a vector of size 768 for each sample in the batch 
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output




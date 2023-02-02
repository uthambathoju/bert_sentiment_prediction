"""
    -Define the data loader , here the dataloader is class bertdataset
"""

import torch
#user-defined
import config
import os
os.chdir('D:/Utham/kaggle/bert_sentiment_prediction/')
          
class BertDataset:
    def __init__(self , review , target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)
    
    def __getitem__(self , item):
        review = str(self.review)
        #remove the space
        review = " ".join(review.split())
        #encode_plus can encode two strings at a time, we have only one in this case 
        inputs = self.tokenizer.encode_plus(
            review , 
            None,
            add_special_tokens=True,
            max_len = self.max_len
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"] 
        #padding
        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        """
            - since we have taken a linear layer with one ouptut so , its torch.float
            - if u plan to have two outputs instead , it depends on what kind of loss fn u use
              so , for cross entropy u use torch.long
        """
        return {
            "ids" : torch.tensor(ids , dtype = torch.long),
            "mask" : torch.tensor(mask , dtype = torch.long),
            "token_type_ids" : torch.tensor(token_type_ids , dtype = torch.long),
            "target" : torch.tensor(self.target[item] , dtype = torch.float)
        } 


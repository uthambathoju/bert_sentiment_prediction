"""
 - this file contains what needed the most like the configurations
 - below parameters are needed during training and prediction, its better to just happen at one place
"""
import transformers
import os
os.chdir('D:/Utham/kaggle/bert_sentiment_prediction/')

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10 
ACCUMULATON = 2
BERT_PATH = './input/bert_base_uncased'
MODEL_PATH = 'model.bin' 
TRAINING_FILE = './input/imdb.csv'
"""
    - we are training only one model, so we need only one kind of tokenizer 
    - if u have multiple tokenizers , watch for tokenizer dispatcher , model dispatcher python files from other 
      projects which will give u a better way to evaluate different types of models
"""
TOKENIZER = transformers.BertTokenizer.from_pretrained(
                                                    BERT_PATH,
                                                    do_lower_case = True
                                                    )

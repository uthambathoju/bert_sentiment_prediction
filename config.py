import os
import tokenizers

os.chdir('D:/Utham/kaggle/text_extraction_from_corpus/input/bert_base_uncased/')
#sequence length
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
BERT_PATH = "D:/Utham/kaggle/text_extraction_from_corpus/input/bert_base_uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "D:/Utham/kaggle/text_extraction_from_corpus/input/train.csv"
#tokenizers library is much faster than transformers berttokenizer with extra features
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH , "vocab.txt")
)

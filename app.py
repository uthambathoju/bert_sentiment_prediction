"""
 - serve the model using flask , this will take a sentence from end-uer and predict positive/negative
 - u can also serve using batches but here we are looking at simply predictions for one single sentence

- cache techniques
    - 1.predict from cache function
    - 2.LRU(Least Recently Used) cache
        - LRU will keep in its memory only the most recently used arguments so its very easy to use
          LRU cache in python.
    -3. using joblib(favorite for abhishek thakur)
        - reason be'coz first two techniques will use the memory if the inputs size increases RAM will explode
          at some point and crash.
        - instead of doing that u can save it on disk, u can use jobilb which has a memory module.
"""

import torch
import flask
import torch.nn as nn
from flask import Flask
from flask import requests
from model import BERTBaseUncased

#user-defined
import config

#functools is used for higher order function and functions that act or return other functions
import functools

import joblib

app = Flask(__name__)

MODEL = None
DEVICE = "cuda"
PREDICTION_DICT = {}
memory = joblib.Memory("./input/" , verbose=0)

def predict_from_cache(sentence):
    if sentence in PREDICTION_DICT:
        return PREDICTION_DICT[sentence]
    else:
        result = sentence_prediction(sentence)
        PREDICTION_DICT[sentence] = result
        return result


#maxsize is how many requests u want to be saved
#@functools.lru_cache(maxsize=128) - second technique
@memory.cache
def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_length = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    #data loader returns batches so we need to add unsqueeze() which adds one more dimension
    #so u r batch size is 1 now
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long) 

    outputs = MODEL(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


#create end-point using flask
@app.route("/predict")
def predict():
    response = {}
    sentence = request.args.get("sentence")
    #belowline of code gives only positive prediction , be'coz we trained it only one class/output
    positive_prediction = sentence_prediction(sentence, model=MODEL)
    #positive_prediction = predict_from_cache(sentence)

    negative_prediction = 1 - positive_prediction
    response["response"] = {
        'positive': str(positive_prediction),
        'negative': str(negative_prediction),
        'sentence': str(sentence)
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()
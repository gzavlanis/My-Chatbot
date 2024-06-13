import numpy as np
import pickle
import json
from flask import Flask, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

model = load_model('../chatbot.keras') # load the trained model

with open('../tokenizer.pickle', 'rb') as handle: # load tokenizer object
    tokenizer = pickle.load(handle)

with open('../label_encoder.pickle', 'rb') as enc: # load the label encoder object
    lbl_encoder = pickle.load(enc)

with open("../data/intents.json") as file:
    data = json.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Welcome to our chatbot!</h1><h4>Application running</h4>"

@app.route("/get", methods = ["POST"])
def chatbot_response():
    res = getResponse(data)
    return res

def getResponse(data):
    max_len = 20
    msg = request.form["msg"]
    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([msg]), truncating = 'post', maxlen = max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            result = np.random.choice(i['responses'])
            break

    return result

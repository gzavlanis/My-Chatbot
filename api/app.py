import random
import numpy as np
import pickle
import json
import nltk
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

Lemmatizer = WordNetLemmatizer()

model = load_model("../chatbot.keras")
intents = json.loads(open("../data/intents.json").read())
# to be continued...
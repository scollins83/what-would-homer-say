from flask import Flask, request
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)
global model, graph
model, graph = init()

@app.route
from flask import Flask, jsonify, request, send_file
import numpy as np
import torch

from ui.algo import Eval

app = Flask(__name__)
algo = Eval()

@app.route('/')
def hello_world():
        return "API"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        shape = algo.create(request.form["text"])
        np.save('./tmp/shape.npy', shape.detach().numpy())
        return send_file('./tmp/shape.npy')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
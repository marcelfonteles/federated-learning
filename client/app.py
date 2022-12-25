from flask import Flask, request
import pickle
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import copy
import os
import json

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from src.models import CNNMnist
from src.datasets import get_dataset
from src.update import LocalUpdate
from src.utils import logging


# Subscribing to server
client_id = -1
response = requests.post('http://localhost:3000/subscribe')
client_id = response.json()['id']


# Initialization: get the last version of global model
response = requests.post('http://localhost:3000/get_model')
local_weights = pickle.loads(response.content)
local_model = CNNMnist()
local_model.load_state_dict(local_weights)
local_model.train()


# Initialization: randomly select the data from dataset for this client
n_clients = 20
headers = {'Content-Type': 'application/json'}
response = requests.post('http://localhost:3000/get_data', json={"client_id": client_id}, headers=headers)
user_group = response.json()['dict_user']
user_group = set(json.loads(user_group))
train_dataset, test_dataset = get_dataset(n_clients)


# Log file path
dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, 'logs/' + str(client_id) + '.log')
logging(f'| #### Starting a new client #### |', True, log_path)

# Flask App
app = Flask(__name__)


# Configuring job to verify if they need to train
def train():
    global local_model, local_weights
    headers = {'Content-Type': 'application/json'}
    data = {"client_id": client_id}
    res = requests.post('http://localhost:3000/get_clients_to_train', json=data, headers=headers)
    res_json = res.json()
    must_train = res_json['train']
    logging(f'| client_id: {client_id}, must_train: {must_train} |', True, log_path)
    if must_train:
        global_epoch = res_json['epoch']
        # Get the newest global model
        res = requests.post('http://localhost:3000/get_model')
        local_weights = pickle.loads(res.content)
        local_model = CNNMnist()
        local_model.load_state_dict(local_weights)
        local_model.train()

        # Get the training parameters (batch_size, n_epochs)
        local = LocalUpdate(dataset=train_dataset, idxs=user_group)
        w, loss = local.update_weights(model=copy.deepcopy(local_model), global_round=global_epoch, client=client_id, log_path=log_path)

        # New local model with updated weights
        local_model.load_state_dict(w)

        # Send new local model to server
        serialized = pickle.dumps(w)
        url = 'http://localhost:3000/send_model'
        headers = {"Client-Id": str(client_id), 'Content-Type': 'application/octet-stream'}
        requests.post(url, data=serialized, headers=headers)


scheduler = BackgroundScheduler()
scheduler.add_job(func=train, trigger="interval", seconds=60)
scheduler.start()


# Healthcheck route
@app.route("/", methods=['GET', 'POST'])
def home():
    return {"message": "Client is running."}


# This route is used to trigger the client to train their local model.
# When called this endpoint we must send n_epochs and batch_size.
# The response is the update weights for local model.
@app.route("/train", methods=['POST'])
def train():
    return {"hello": "world", "array": [1, 2, 3], "nested": {"again": 1}}


# This route can be used to test to model prediction in client
# Todo: Add a simple interface so the user can test.
@app.route("/predict", methods=['POST'])
def predict():
    image = request.files['image']
    img = Image.open(image)
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    pred = torch.argmax(local_model(img_tensor))
    return {"prediction": pred.item()}

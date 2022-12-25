from flask import Flask, Response
import pickle
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import copy

from src.models import CNNMnist
from src.datasets import get_dataset
from src.update import LocalUpdate


####
# Steps
# initialization => get the global model from the server
# randomly select data from dataset
# this selection must be determined from how many clients we have.
####

# Initialization: get the last version of global model
response = requests.post('http://localhost:3000/get_model')
local_weights = pickle.loads(response.content)
local_model = CNNMnist()
local_model.load_state_dict(local_weights)
local_model.train()

# Initialization: randomly select the data from dataset for this client
n_users = 1
train_dataset, test_dataset, user_group = get_dataset(n_users)

# Subscribing to server
client_id = -1
response = requests.post('http://localhost:3000/subscribe')
client_id = response.json()['id']

app = Flask(__name__)


# Configuring job to verify if they need to train
def train():
    response = requests.post('http://localhost:3000/get_clients_to_train')
    clients_to_train = response.json()['clients']
    print(f'client_id: {client_id}, clients_to_train: {clients_to_train}')
    if clients_to_train.count(client_id):
        # Get the newest global model
        reponse = requests.post('http://localhost:3000/get_model')
        local_weights = pickle.loads(reponse.content)
        local_model = CNNMnist()
        local_model.load_state_dict(local_weights)
        local_model.train()

        # Get the training parameters (batch_size, n_epochs)
        local = LocalUpdate(dataset=train_dataset, idxs=user_group)
        w, loss = local.update_weights(model=copy.deepcopy(local_model), global_round=1, client=client_id)

        # New local model with updated weights
        local_model.load_state_dict(w)

        # Send new local model to server
        # model, client_id
        serialized = pickle.dumps(w)
        url = 'http://localhost:3000/send_model'
        headers = {"Client-Id": '10', 'Content-Type': 'application/octet-stream'}
        requests.post(url, data=serialized, headers=headers)


# scheduler = BackgroundScheduler()
# scheduler.add_job(func=train, trigger="interval", seconds=10)
# scheduler.start()
train()

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
    return {"hello": "world", "array": [1, 2, 3], "nested": {"again": 1}}

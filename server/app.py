from flask import Flask, Response, request
import pickle
import numpy as np
from src.models import CNNMnist
from src.utils import average_weights, test_inference
from src.datasets import get_test_dataset, get_user_group


global_model = CNNMnist()
global_model.train()

# Total of clients
n_clients = 20

# List of client ids
clients = []

# Fraction of clients that must train local model on each epoch
frac_to_train = 0.50
n_clients_to_train = frac_to_train * n_clients

# List of clients that must train local model on each epoch
clients_to_train = []

# List of weights after local training on each client
clients_weights = []

# Global Epochs
n_epochs = 10
current_epoch = 1

# Client Data
dict_users = {}

# Flask App
app = Flask(__name__)


# Healthcheck route
@app.route("/", methods=['GET', 'POST'])
def home():
    return {"message": "Server is running."}


# Client will consume this route to get data from dataset.
# In real life all clients will have their own data, but in this test we must
#    guarantee that all clients have different data
@app.route("/get_data", methods=['POST'])
def get_data():
    data = request.json
    client_id = data['client_id']
    dict_user = get_user_group(n_clients, dict_users)
    dict_users[client_id] = dict_user

    return {"dict_user": str(list(dict_user))}


# Client will consume this route to check if they need to train theirs local model
@app.route("/get_clients_to_train", methods=['POST'])
def get_clients_to_train():
    data = request.json
    client_id = data['client_id']
    if clients_to_train.count(client_id):
        clients_to_train.remove(client_id)
        return {"train": True, "epoch": current_epoch}
    else:
        return {"train": False}


# Clients will consume this route to obtain the last version of the Global Model.
# This endpoint will send the weights in a binary form
@app.route("/get_model", methods=['POST'])
def get_model():
    global_weights = global_model.state_dict()

    serialized = pickle.dumps(global_weights)
    response = Response(serialized)
    response.headers['Content-Type'] = 'application/octet-stream'
    return response


# Clients will consume this route to send theirs updated local model.
# This endpoint will execute the fedAvg algorithm and return the new global model
@app.route("/send_model", methods=['POST'])
def send_model():
    global clients_to_train, clients_weights, current_epoch
    weights = pickle.loads(request.data)
    clients_weights.append(weights)
    if len(clients_weights) == n_clients_to_train:
        # fedAvg
        # update global weights
        print(f'Fazendo média | epoch {current_epoch}')
        global_weights = average_weights(clients_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # random select new clients
        if current_epoch < n_epochs:
            print(f'Escolhendo novos clientes | epoch {current_epoch}')
            clients_weights = []
            m = max(int(frac_to_train * n_clients), 1)
            clients_to_train = np.random.choice(range(n_clients), m, replace=False).tolist()
            current_epoch += 1
        else:
            print('training is complete')
            test_dataset = get_test_dataset()
            test_acc, test_loss = test_inference(global_model, test_dataset)

            print(f' \n Results after {current_epoch} global rounds of training:')
            # print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    return {}


# This route will subscribe the client
@app.route("/subscribe", methods=['POST'])
def subscribe():
    global clients_to_train
    if len(clients) < n_clients:
        client_id = len(clients)
        clients.append(client_id)
        if len(clients) == n_clients:
            m = max(int(frac_to_train * n_clients), 1)
            clients_to_train = np.random.choice(range(n_clients), m, replace=False).tolist()
        return {"id": client_id}
    else:
        return {"id": None}


# This route will start training process
@app.route("/start_training", methods=['POST'])
def start_training():
    return {"hello": "world", "array": [1, 2, 3], "nested": {"again": 1}}

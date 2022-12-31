import time

from flask import Flask, Response, request
import pickle
import numpy as np
from datetime import datetime
import pymongo
import json
from src.models import CNNMnist, CNNCifar
from src.utils import average_weights, test_inference
from src.datasets import get_test_dataset, get_user_group
from src.database import get_database


dataset = 'mnist'  # or mnist
if dataset == 'mnist':
    # Training for MNIST
    global_model = CNNMnist()
else:
    # Training for CIFAR10
    global_model = CNNCifar()
global_model.train()

'''#########################'''
'''Turning the server stateless'''

# MongoDB Connection
training_db = get_database()
global_models_table = training_db.global_models
records = [record for record in global_models_table.find().sort('createdAt', pymongo.DESCENDING)]
if len(records) == 0:  # create a new record on db
    global_weights = global_model.state_dict()
    serialized = pickle.dumps(global_weights)
    global_models_table.insert_one({
        'id': 1,
        'dataset': dataset,
        'serialized': serialized,
        'totalEpochs': 1,
        'currentEpoch': 1,
        'isTraining': True,
        'createdAt': datetime.now(),
        'updatedAt': datetime.now(),
    })
else:  # load or create from db
    record = records[0]  # ordered by createdAt DESCENDING, position 0 is the newest one.
    if record['isTraining']:  # load binary from db
        serialized = record['serialized']
        weights = pickle.loads(serialized)
        global_model.load_state_dict(weights)
    else:  # start a new model
        global_weights = global_model.state_dict()
        serialized = pickle.dumps(global_weights)
        global_models_table.insert_one({
            'id': record['id'] + 1,
            'dataset': dataset,
            'serialized': serialized,
            'totalEpochs': 5,
            'currentEpoch': 1,
            'isTraining': True,
            'createdAt': datetime.now(),
            'updatedAt': datetime.now(),
        })


# Total of clients
n_clients = 10

# Fraction of clients that must train local model on each epoch
frac_to_train = 0.50
n_clients_to_train = frac_to_train * n_clients

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
    # get global model
    records = [record for record in training_db.global_models.find().sort('createdAt', pymongo.DESCENDING)]
    if len(records) == 0:
        raise 'No Global Model Available'
    else:
        global_model_record = records[0]
    # get all clients
    clients = [record for record in training_db.clients.find({'globalModelId': global_model_record['id']}).sort('id', pymongo.ASCENDING)]
    client = training_db.clients.find_one({'globalModelId': global_model_record['id'], 'id': data['client_id']})
    dict_users = {}
    for c in clients:
        if c['datasetIndexes'] != []:
            dict_users[c['id']] = set(json.loads(c['datasetIndexes']))

    dict_user = get_user_group(dataset, n_clients, dict_users)
    training_db.clients.update_one(
        {'globalModelId': global_model_record['id'], 'id': client['id']},
        {'$set': {'datasetIndexes': str(list(dict_user)), 'updatedAt': datetime.now()}
    })
    return {"dict_user": str(list(dict_user))}


# Client will consume this route to check if they need to train theirs local model
@app.route("/get_clients_to_train", methods=['POST'])
def get_clients_to_train():
    data = request.json
    client_id = data['client_id']
    records = [record for record in training_db.global_models.find().sort('id', pymongo.DESCENDING)]
    if len(records) == 0:
        raise 'No Global Model Available'
    else:
        global_model_record = records[0]

    records = [record for record in training_db.training_clients.find({'globalModelId': global_model_record['id']}).sort('id', pymongo.DESCENDING)]
    if len(records) == 0 or global_model_record['isTraining'] == False:
        clients_to_train = []
    else:
        clients_to_train = records[0]['clients']

    records = [record for record in training_db.global_models.find().sort('id', pymongo.DESCENDING)]
    if len(records) == 0:
        current_epoch = 0
    else:
        current_epoch = records[0]['currentEpoch']

    if clients_to_train.count(client_id):
        clients_to_train.remove(client_id)
        return {"train": True, "epoch": current_epoch}
    else:
        return {"train": False}


# Clients will consume this route to obtain the last version of the Global Model.
# This endpoint will send the weights in a binary form
@app.route("/get_model", methods=['POST'])
def get_model():
    records = [record for record in training_db.global_models.find().sort('id', pymongo.DESCENDING)]
    if len(records) == 0:
        global_weights = global_model.state_dict()
        serialized = pickle.dumps(global_weights)
    else:
        serialized = records[0]['serialized']

    response = Response(serialized)
    response.headers['Content-Type'] = 'application/octet-stream'
    return response


# Clients will consume this route to send theirs updated local model.
# This endpoint will execute the fedAvg algorithm and return the new global model
@app.route("/send_model", methods=['POST'])
def send_model():
    # Insert model data in local model table
    client_id = int(request.headers['Client-Id'])

    records = [record for record in training_db.local_models.find().sort('id', pymongo.DESCENDING)]
    if len(records) == 0:
        local_model_id = 1
    else:
        local_model_id = records[0]['id'] + 1

    records = [record for record in training_db.global_models.find().sort('id', pymongo.DESCENDING)]
    if len(records) == 0:
        raise 'No Global Model Available'
    else:
        global_model_record = records[0]

    find_query = {
        'clientId': client_id,
        'globalModelId': global_model_record['id'],
    }
    records = [record for record in training_db.local_models.find(find_query).sort('id', pymongo.DESCENDING)]
    if len(records) == 0:
        training_db.local_models.insert_one({
            'id': local_model_id,
            'clientId': client_id,
            'globalModelId': global_model_record['id'],
            'model': request.data,
            'epoch': global_model_record['currentEpoch'],
            'createdAt': datetime.now(),
            'updatedAt': datetime.now(),
        })
    else:
        training_db.local_models.update_one(
            {'clientId': client_id, 'globalModelId': global_model_record['id']},
            {'$set': {'model': request.data, 'epoch': global_model_record['currentEpoch'], 'updatedAt': datetime.now()}}
        )

    clients_weights = []
    find_query = {
        'epoch': global_model_record['currentEpoch'],
        'globalModelId': global_model_record['id'],
    }
    records = [record for record in training_db.local_models.find(find_query).sort('id', pymongo.DESCENDING)]
    for record in records:
        weights = pickle.loads(record['model'])
        clients_weights.append(weights)

    if len(clients_weights) == n_clients_to_train:
        # fedAvg
        # update global weights
        current_epoch = global_model_record['currentEpoch']
        n_epochs = global_model_record['totalEpochs']
        print(f'Fazendo média | epoch {current_epoch}')
        global_weights = average_weights(clients_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # random select new clients
        if current_epoch < n_epochs:
            print(f'Escolhendo novos clientes | epoch {current_epoch}')
            m = max(int(frac_to_train * n_clients), 1)
            clients_id_to_train = np.random.choice(range(n_clients), m, replace=False).tolist()

            records = [record for record in training_db.training_clients.find({'globalModelId': global_model_record['id']}).sort('createdAt', pymongo.DESCENDING)]
            if len(records) == 0:
                training_clients_id = 1
            else:
                training_clients_id = records[0]['id'] + 1

            training_db.training_clients.insert_one({
                'id': training_clients_id,
                'clients': clients_id_to_train,
                'globalModelId': global_model_record['id'],
                'currentEpoch': global_model_record['currentEpoch'] + 1,
                'createdAt': datetime.now(),
                'updatedAt': datetime.now(),
            })

            training_db.global_models.update_one(
                {'id': global_model_record['id']},
                {'$set': {'currentEpoch': global_model_record['currentEpoch'] + 1, 'updatedAt': datetime.now()}
                 })

        else:
            training_db.global_models.update_one(
                {'id': global_model_record['id']},
                {'$set': {'isTraining': False, 'updatedAt': datetime.now()}
            })
            print('training is complete')
            test_dataset = get_test_dataset(dataset)
            test_acc, test_loss = test_inference(global_model, test_dataset)

            print(f' \n Results after {current_epoch} global rounds of training:')
            # print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    return {}


# This route will subscribe the client
@app.route("/subscribe", methods=['POST'])
def subscribe():
    # get global model
    records = [record for record in training_db.global_models.find().sort('createdAt', pymongo.DESCENDING)]
    if len(records) == 0:
        raise 'No Global Model Available'
    else:
        global_model_record = records[0]

    # get clients of this model
    clients = [record for record in training_db.clients.find({'globalModelId': global_model_record['id']}).sort('id', pymongo.ASCENDING)]

    if len(clients) < n_clients:
        client_id = len(clients)
        client = {
            'id': client_id,
            'globalModelId': global_model_record['id'],
            'datasetIndexes': [],
            'createdAt': datetime.now(),
            'updatedAt': datetime.now(),
        }
        training_db.clients.insert_one(client)
        clients.append(client)
        if len(clients) == n_clients:
            m = max(int(frac_to_train * n_clients), 1)
            clients_id_to_train = np.random.choice(range(n_clients), m, replace=False).tolist()
            records = [record for record in training_db.training_clients.find().sort('createdAt', pymongo.DESCENDING)]
            if len(records) == 0:
                training_clients_id = 1
            else:
                training_clients_id = records[0]['id'] + 1

            records = [record for record in training_db.global_models.find().sort('createdAt', pymongo.DESCENDING)]
            if len(records) == 0:
                raise 'No Global Model Available'
            else:
                global_model_id = records[0]['id']

            training_db.training_clients.insert_one({
                'id': training_clients_id,
                'clients': clients_id_to_train,
                'globalModelId': global_model_id,
                'currentEpoch': 1,
                'createdAt': datetime.now(),
                'updatedAt': datetime.now(),
            })
        return {"id": client_id}
    else:
        return {"id": None}


# This route will start training process
@app.route("/start_training", methods=['POST'])
def start_training():
    return {"hello": "world", "array": [1, 2, 3], "nested": {"again": 1}}

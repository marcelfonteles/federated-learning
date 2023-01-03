from pymongo import MongoClient, ASCENDING, DESCENDING
import os


def get_database():
    try:
        client = MongoClient(os.environ['MONGODB_URL'], 27017, username='root', password='password')
    except Exception as e:
        client = MongoClient('localhost', 27017, username='root', password='password')
    return client.training


def get_last_global_model():
    client = get_database()
    training = client.training
    global_models = training.global_models
    records = [record for record in global_models.find().sort('createdAt', ASCENDING)]
    client.close()

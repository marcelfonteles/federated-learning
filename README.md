# Federated Learning

### Requirements
Install all the packages from requirments.txt

- flask
- torch
- torchvision
- numpy
- tensorboardx
- matplotlib
- apscheduler
- Pillow
```
pip3 install -r client/requirements.txt
pip3 install -r server/requirements.txt
```

### Models and Dataset
Currently, this project uses CNN with 2 layers and two types of dataset: MNIST and CIFAR10. We use `SGD` as optimizer

### How does this project works
We have two mainly pieces of code, one represents the server and other represents the client (the entity who owns 
the data). So firstly we must start one server and after that we initialize multiple clients (this will 
represent multiple users).

After that, all client subscribe to the server and get a subset of the dataset, this is one way to guarantee that all
users have different data. The server do not see the data and do not train any model with the data. The server 
only randomly choose a subset of dataset and send this information to the user.

For default, we will use 20 clients and the dataset will be divided equally to all the users.

After the division of the dataset, the server will randomly choose 10 clients at each epoch to train their local model.
On the client, the local model will be trained using CNN and 10 epochs, after the training each local model will be
sent to the server and then the fedAvg algorithm will be executed to generate a new global model. 
This process will repeat 10 times and after that we will have a global model ready to use for predictions for new data. 

### Running the experiments
First start the server
```
flask --app server/app.py run -h 0.0.0.0 -p 3000
```

Then run all the clients by executing the `start.sh` script
````
sh start.sh
````
By default, this will start 20 clients on differents ports between 3001 and 3021.
At each `global epoch` the server will randomly choose 10 clients the train theirs local models
 
After the training you can use this route `localhost:<3001~3021>/predict` to predict

### Running the experiments (with Docker)
Build server docker image
```
docker build -t server -f Dockerfile .
```

Build client docker image
```
docker build -t client -f Dockerfile .
```


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

### Running the experiments
First start the server
```
flask --app server/app.py -h 0.0.0.0 -p 3000
```

Then run all the clients by executing the `start.sh` script
````
sh start.sh
````
By default this will start 20 clients on differents ports between 3001 and 3021.
At each `global epoch` the server will randomly choose 10 clients the train theirs local models
 
After the training you can use this route `localhost:<3001~3021>/predict` to predict
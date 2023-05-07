# 17-645 (S'23) ML in Production: Group Project
Team 13: _The Tensor Titans_

clone:
```
$ git clone https://github.com/cmu-seai/group-project-s23-the-tensor-titans.git
$ cd group-project-s23-the-tensor-titans
```

create & activate virtual env:
```
$ python3 -m venv env
$ source env/bin/activate
```

install dependencies:
```
$ pip install -r requirements.txt
```

set env variable:
```
$ export RECOM_MODEL_PATH='./MLModels/model.bin'
```

start the flask server, use:
```
python3 flask_app.py
```
instead of:
```
flask run
```

## Testing
* install test packages:
```
$ python3 -m pip install pytest-cov
$ python3 -m pip install pytest-mock
```

* run tests and report coverage
```
$  pytest --cov-report term --cov=online_eval --cov=MLModels --cov=flask_app --cov=kafkaConsumer test/
```

## Docker Containerization
docker has alreay been installed on our VM. To check current built docker images, use:
```
$ docker images
```
Our current service contains two containers built from the same docker image "m3:testv1". Both movie_recommend_service listen on the container port 8081.

To start the service, run:
```
$ cd docker
$ docker-compose up
```
Inside the docker-compose.yaml file, we define the configuration of the service. The request to the container a will be mapped through **local network port 8001 to container port 8081**. Similarly, the request to the container b will be mapped through **local network port 8002 to container port 8081**.

After the containers are up and running, start the load balancer service:
```
$ python3 load_balancer.py
```
The load balancer service will be running on port 8081 of local network. **Even** user_id will be directed to container a while **odd** user_id will be directed to container b.

For the container logs, please write all your log files to the folder **/var** which is in the container file system when developing your code. All such log files will appear in the local folders **/logvolume01** and **/logvolume02**, respectively.

Use
```
$ cd docker
$ docker-compose down
```
to stop and remove the containers created above.
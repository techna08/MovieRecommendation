'''
Reference:
https://github.com/mlip-cmu/s2023/blob/main/recitations/Recitation%205/Docker%20Demo%20Code/load_balancer_demo/load_balancer.py
'''
import sys
import os
from flask import Flask
from flask import request, make_response
import requests
from MLModels import predict

app = Flask('load-balancer-server')

container_ports = [
    'http://localhost:8001',
    'http://localhost:8002',
]
health_check_ip = [
    '0.0.0.0 8001',
    '0.0.0.0 8002',
]

# create two log directory if not exists on the host machine
logDir1 = "./logvolume_baseline"
logDir2 = "./logvolume_svd"
if not os.path.exists(logDir1):
    os.makedirs(logDir1)
if not os.path.exists(logDir2):
    os.makedirs(logDir2)


def checkHealth(ip_addr):
    return os.system('nc -vz '+ip_addr) == 0

@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    # add health check
    A_server_up = checkHealth(health_check_ip[0])
    B_server_up = checkHealth(health_check_ip[1])

    # load balancing
    if not B_server_up and A_server_up:
    	target_url = container_ports[0] + '/recommend/' + userid
    elif B_server_up and not A_server_up:
        target_url = container_ports[1] + '/recommend/' + userid
    elif A_server_up and B_server_up:
        # NOTE(Shih-Lun): only direct 10 percent of traffic
        #                 to baseline model as it likely underperforms
        if userid[-1] == "0":
            target_url = container_ports[0] + '/recommend/' + userid
        else:
            target_url = container_ports[1] + '/recommend/' + userid
    else:
        return f"Service Unavailable", 503
    
    response = requests.request(method='GET', url=target_url)

    return response.text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=False)
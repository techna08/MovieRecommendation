import os, pickle, random 
from flask import Response,Flask, request, jsonify 
import predict
from kafka import KafkaConsumer
import prometheus_client
from prometheus_client.core import CollectorRegistry
from prometheus_client import Summary, Counter, Histogram, Gauge
from online_eval.online_evaluate import compute_online_metrics
import sys
import threading
import json
import time



logFile_root = "/var"
if not os.path.exists(logFile_root):
    os.makedirs(logFile_root)
import re

app = Flask(__name__)

topic ='movielog13'


graphs = {}
graphs['c'] = Counter(
    'request_count', 'Recommendation Request Count',
    ['http_status']
)
graphs['h'] = Histogram('request_latency_seconds', 'Request latency')
graphs['g'] =  last_hr_rating = Gauge('last_hour_rating', 'Last Hour Rating')
@app.route("/metrics")
def requests_count():
    res = []
    for k,v in graphs.items():
        res.append(prometheus_client.generate_latest(v))
    return Response(res, mimetype="text/plain")

def background_job():
    # Code to be executed periodically goes here
    print('Background job running...')
    consumer = KafkaConsumer(
        topic, #topic here
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        group_id=topic, #group ID here
        enable_auto_commit=True,
        auto_commit_interval_ms=1000
    )
    for message in consumer:
        event = message.value.decode('utf-8')
        values = event.split(',')

        if 'recommendation request' in values[2]:
            #print(values)
            status = values[3].strip().split(" ")[1]
            graphs['c'].labels(status).inc()
            #print(status)

            time_taken = float(values[-1].strip().split(" ")[0])
            graphs['h'].observe(time_taken / 1000)

def background_job2():
    while True:
        result=compute_online_metrics(True)
        
        last_hr_rating = result['last_hr_rating']
        timestamp = int(time.time())
        graphs['g'].set(last_hr_rating)
        print("Result is {}", result)
        time.sleep(3600)

def start_background_job():
    thread = threading.Timer(10.0, background_job)
    thread.daemon = True
    thread.start()
def start_background_job2():
    thread = threading.Timer(10.0, background_job2)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    start_background_job()
    start_background_job2()
    app.run(port=8765, host='0.0.0.0')
    

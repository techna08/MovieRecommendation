from os import path
import sys, os
from datetime import datetime
from json import dumps, loads
from time import sleep
from random import randint
import numpy as np
import re
# ssh -o ServerAliveInterval=60 -L 9092:localhost:9092 tunnel@128.2.24.106 -NTf
# ssh -o ServerAliveInterval=60 -L 9092:localhost:9092 tunnel@128.2.204.215 -NTf
# scp aarushis@128.2.205.114:/path/to/file /path/to/destination/on/local/machine
# scp /path/to/local/file aarushis@128.2.205.114:/home/aarushis/test

from kafka import KafkaConsumer, KafkaProducer



def getBucket(timestamp):
    # Extract the hour component from the timestamp
    hour = int(timestamp[:2])

    # Define the buckets
    buckets = {
        "0000-0100": range(0, 1),
        "0100-0200": range(1, 2),
        "0200-0300": range(2, 3),
        "0300-0400": range(3, 4),
        "0400-0500": range(4, 5),
        "0500-0600": range(5, 6),
        "0600-0700": range(6, 7),
        "0700-0800": range(7, 8),
        "0800-0900": range(8, 9),
        "0900-1000": range(9, 10),
        "1000-1100": range(10, 11),
        "1100-1200": range(11, 12),
        "1200-1300": range(12, 13),
        "1300-1400": range(13, 14),
        "1400-1500": range(14, 15),
        "1500-1600": range(15, 16),
        "1600-1700": range(16, 17),
        "1700-1800": range(17, 18),
        "1800-1900": range(18, 19),
        "1900-2000": range(19, 20),
        "2000-2100": range(20, 21),
        "2100-2200": range(21, 22),
        "2200-2300": range(22, 23),
        "2300-0000": range(23, 24)
    }
    for bucket, hour_range in buckets.items():
        if hour in hour_range:
            return bucket

def getPath(message):
    message_date = message.split(",")[0].split("T")[0]
        #checking message date
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if not re.match(pattern, message_date):
        raise Exception(f'The date string {message_date} does not match the format YYYY-MM-DD')
    message_time = message.split(",")[0].split("T")[1]
    bucket=getBucket(message_time)
    newPath=f"{message_date}/{bucket}"
    if not os.path.exists(newPath):
        os.makedirs(newPath)
    rating_path = f"{newPath}/ratings.txt"
    mpg_path = f"{newPath}/mpg.txt"
    recom_path = f"{newPath}/recommendations.txt"
    return [rating_path,mpg_path,recom_path]

def process_messages(consumer):
    print('Reading Kafka Broker')
    for message in consumer:
        message = message.value.decode()
        # Extract the date from the message
        try:
            paths=getPath(message)
            # Default message.value type is bytes!
            if 'recommendation request' in message:
                with open(paths[2], "a") as f:
                    f.write(message + "\n")
            elif not message.endswith('.mpg'):
                with open(paths[0], "a") as f:
                    f.write(message + "\n")
            elif message.endswith('.mpg'):
                #user = message.split(",")[1]
                #movie = message.split("/")[-2]
                #fo.write(f"{user} {movie}\n")
                with open(paths[1], "a") as f:
                    f.write(message + "\n")
            os.system(f"echo {message} >> kafka_log.csv")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    topic = 'movielog13'
    consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['localhost:9092'],
    # Read from the start of the topic; Default is latest
    #auto_offset_reset='earliest',
    auto_offset_reset='latest',
    # group_id='team13',
    # Commit that an offset has been read
    enable_auto_commit=True,
    # How often to tell Kafka, an offset has been read
    auto_commit_interval_ms=1000
    )   
    process_messages(consumer)
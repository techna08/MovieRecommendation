import sys
import os
import logging
import requests
from io import StringIO
sys.path.append(os.path.dirname(__file__) + "/../../")

from flask_app import recommend

def test_flask_app():
    url = "http://128.2.205.114:8082/recommend/470318"
    response = requests.get(url)
    expected=200
    responseCount=len(str(response.content).split(','))
    expectedCount=8
    assert response.status_code == expected
    assert responseCount >= expectedCount

def test_flask_app_invalid():
    #testing for invalid user id
    url = "http://128.2.205.114:8082/recommend/0000"
    response = requests.get(url)
    expected=200
    responseCount=len(str(response.content).split(','))
    expectedCount=8
    assert response.status_code == expected
    assert responseCount >= expectedCount
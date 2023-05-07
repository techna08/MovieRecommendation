import os 
import sys 
from surprise import SVD, BaselineOnly 
# for modoule inport
sys.path.append(os.path.dirname(__file__) + "/../../MLModels/")
from evaluate import * 

import pytest
from pytest_mock import mocker



def test_train_and_eval_accuracy_svd():
    model_type = "SVD"
    assert train_and_eval_accuracy(model_type)[1] == 1.0

def test_train_and_eval_accuracy_baseline():
    model_type = "baseline"
    assert train_and_eval_accuracy(model_type)[1] == 1.0


def test_measure_inference_speed():
    assert measure_inference_speed(get_example_model()) < 0.1


def test_measure_model_size():
    assert measure_model_size(get_example_model()) < 3.4

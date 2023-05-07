import os 
import sys 
sys.path.append(os.path.dirname(__file__) + "/../../MLModels/")
from grid_search import * 

import pytest
from pytest_mock import mocker


def test_grid_search():
    grid_search()
    assert os.path.exists("test/MLModels/test_model_grid_search.bin") == True

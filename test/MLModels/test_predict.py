import pytest
import os 
import sys
import pickle 
sys.path.append(os.path.dirname(__file__) + "/../../")
from MLModels.predict import *

def test_predict_known_user():
	# Test predicting recommendations for a known user
	print(os.getcwd())
	model = pickle.load(open("test/MLModels/predict_test_model.bin","rb"))
	known_users = model["model"].trainset._raw2inner_id_users.keys()
	for user_id in random.sample(list(known_users), 1):
		result = predict(user_id)
	assert isinstance(result, str)
	assert len(result) > 0

def test_predict_unknown_user():
    # Test predicting recommendations for an unknown user
	user_id = "999"
	result = predict(user_id)
	assert isinstance(result, str)
	assert len(result) > 0

def test_predict_invalid_input():
    # Test predicting recommendations with invalid input
	with pytest.raises(TypeError):
		predict() # no arguments provided

	with pytest.raises(TypeError):
		predict(123) # invalid argument type
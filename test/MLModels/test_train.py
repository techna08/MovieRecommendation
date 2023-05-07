import os 
import pickle
import pytest
import sys 
sys.path.append(os.path.dirname(__file__) + "/../../")
from MLModels.train import *

TMP_RATINGS_PATH = ".test_ratings.tmp"
TMP_VIEWS_PATH = ".test_views.tmp"

TEST_RATINGS = [
    "2023-02-07T09:50:59,012345,GET /rate/abcd+bouki+1973=4",
    "2023-02-07T10:00:05,012345,GET /rate/abcd+bye+love+1995=4",
    "2023-02-07T09:59:58,012345,GET /rate/abcd+2013=3",
    "2023-02-07T10:00:31,012345,GET /rate/abcd+kingdom+2012=4",
    "2023-02-07T10:00:01,012345,GET /rate/abcd+point+2008=3",
    "2023-02-07T09:51:16,012345,GET /rate/abcd+good_+the+bad+and+the+ugly+1966=5",
    "2023-02-07T10:00:06,012345,GET /rate/abcd+and+the+chocolate+factory+2005=3",
    "2023-02-07T09:51:10,012345,GET /rate/abcd+comfort+of+strangers+1990=4",
    "2023-02-07T09:50:44,012345,GET /rate/abcd+thieves+2012=3",
    "2023-02-07T09:50:25,012345,GET /rate/abcd+love+2007=3",
    "2023-02-07T09:51:07,012345,GET /rate/abcd+day+of+the+triffids+1962=3",
    "2023-02-07T09:59:37,012345,GET /rate/abcd+me+and+my+gal+1942=3",
    "2023-02-07T09:51:16,012345,GET /rate/abcd+before+chanel+2009=4",
    "2023-02-07T09:59:25,012345,GET /rate/abcd+action+1991=3",
    "2023-02-07T09:59:23,012345,GET /rate/abcd+1996=4",
    "2023-02-07T09:59:16,012345,GET /rate/abcd+potter+and+the+chamber+of+secrets+2002=4"
]

TEST_VIEWS = [
    "2023-02-07T09:50:59,012345,GET /data/abcd+bouki+1973/7.mpg",
    "2023-02-07T10:00:05,012345,GET /data/abcd+bye+love+1995/7.mpg",
    "2023-02-07T09:59:58,012345,GET /data/abcd+2013/7.mpg",
    "2023-02-07T10:00:31,012345,GET /data/abcd+kingdom+2012/7.mpg",
    "2023-02-07T10:00:01,012345,GET /data/abcd+point+2008/7.mpg",
    "2023-02-07T09:51:16,012345,GET /data/abcd+good_+the+bad+and+the+ugly+1966/7.mpg",
    "2023-02-07T10:00:06,012345,GET /data/abcd+and+the+chocolate+factory+2005/7.mpg",
    "2023-02-07T09:51:10,012345,GET /data/abcd+comfort+of+strangers+1990/7.mpg",
    "2023-02-07T09:50:44,012345,GET /data/abcd+thieves+2012/7.mpg",
    "2023-02-07T09:50:25,012345,GET /data/abcd+love+2007/7.mpg",
    "2023-02-07T09:51:07,012345,GET /data/abcd+day+of+the+triffids+1962/7.mpg",
    "2023-02-07T09:59:37,012345,GET /data/abcd+me+and+my+gal+1942/7.mpg",
    "2023-02-07T09:51:16,012345,GET /data/abcd+before+chanel+2009/7.mpg",
    "2023-02-07T09:59:25,012345,GET /data/abcd+action+1991/7.mpg",
    "2023-02-07T09:59:23,012345,GET /data/abcd+1996/7.mpg",
    "2023-02-07T09:59:16,012345,GET /data/abcd+potter+and+the+chamber+of+secrets+2002/7.mpg"
]


@pytest.fixture
def tmp_ratings_file():
    with open(TMP_RATINGS_PATH, "w") as f:
        for l in TEST_RATINGS:
            f.write(l + "\n")

    yield TMP_RATINGS_PATH

    os.remove(TMP_RATINGS_PATH)


@pytest.fixture
def tmp_views_file():
    with open(TMP_VIEWS_PATH, "w") as f:
        for l in TEST_VIEWS:
            f.write(l + "\n")

    yield TMP_VIEWS_PATH

    os.remove(TMP_VIEWS_PATH)

# Test the function with the "SVD" model type and valid train and view history paths:
def test_train_and_dump_model_with_svd(tmp_ratings_file,tmp_views_file):
    model_type = "SVD"
    train_data_path = tmp_ratings_file
    view_history_path = tmp_views_file
    model_out_path = "test_model.bin"

    train_and_dump_model(model_type, train_data_path, view_history_path, model_out_path)

    # Verify that the model file was created
    assert os.path.exists(model_out_path)

    # Verify that the model can be loaded and used for prediction
    loaded_model = pickle.load(open(model_out_path, "rb"))["model"]
    assert isinstance(loaded_model, SVD)
    os.remove(model_out_path)

#Test the function with the "baseline" model type and valid train and view history paths:
def test_train_and_dump_model_with_baseline(tmp_ratings_file,tmp_views_file):
    model_type = "baseline"
    train_data_path = tmp_ratings_file
    view_history_path = tmp_views_file
    model_out_path = "test_model.bin"

    train_and_dump_model(model_type, train_data_path, view_history_path, model_out_path)

    # Verify that the model file was created
    assert os.path.exists(model_out_path)

    #do os remove the model_out_path

    # Verify that the model can be loaded and used for prediction
    loaded_model = pickle.load(open(model_out_path, "rb"))["model"]
    assert isinstance(loaded_model, BaselineOnly)
    os.remove(model_out_path)

#Test the function with an invalid model type
def test_train_and_dump_model_with_invalid_model_type(tmp_ratings_file,tmp_views_file):
    model_type = "invalid_model_type"
    train_data_path = tmp_ratings_file
    view_history_path = tmp_views_file
    model_out_path = "test_model.bin"

    with pytest.raises(ValueError):
        train_and_dump_model(model_type, train_data_path, view_history_path, model_out_path)

    # Verify that the model file was not created
    assert not os.path.exists(model_out_path)

#Test the function with an invalid train data path
def test_train_and_dump_model_with_invalid_train_data_path(tmp_views_file):
    model_type = "SVD"
    train_data_path = "invalid_path.csv"
    view_history_path = tmp_views_file
    model_out_path = "test_model.bin"

    with pytest.raises(FileNotFoundError):
        train_and_dump_model(model_type, train_data_path, view_history_path, model_out_path)

    # Verify that the model file was not created
    assert not os.path.exists(model_out_path)

#Test the function with an invalid view history path:
def test_train_and_dump_model_with_invalid_view_history_path(tmp_ratings_file):
    model_type = "SVD"
    train_data_path = tmp_ratings_file
    view_history_path = "invalid_path.csv"
    model_out_path = "test_model.bin"

    with pytest.raises(FileNotFoundError):
        train_and_dump_model(model_type, train_data_path, view_history_path, model_out_path)

    # Verify that the model file was not created
    assert not os.path.exists(model_out_path)
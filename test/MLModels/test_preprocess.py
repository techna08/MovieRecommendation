import os 
import sys 
from surprise import Dataset, Reader 
import pandas as pd 
# for modoule inport
sys.path.append(os.path.dirname(__file__) + "/../../MLModels/")
from preprocess import * 

import pytest
from pytest_mock import mocker



def test_get_rating_stats():
    rating_stats_filepath = "MLModels/test_ratings.txt"
    example_dict = {'n_ratings': {'the+shining+1980': 1, 'a+woman+under+the+influence+1974': 1, 'operation+y+and+other+shuriks+adventures+1965': 1}, 'rating_t_scores': {'a+woman+under+the+influence+1974': -1.224744871391589, 'operation+y+and+other+shuriks+adventures+1965': 0.0, 'the+shining+1980': 1.224744871391589}}
    assert get_rating_stats(rating_stats_filepath) == example_dict


# TO DO: check the reference format of .mpg file
def test_get_view_history_stats():
    history_stats_filepath = "MLModels/test_mpg.txt"
    example_dict = {
        'views_t_scores': {
            'abcd+before+chanel+2009': 0.0, 
            'abcd+day+of+the+triffids+1962': 0.0, 
            'abcd+love+2007': 0.0, 
            'abcd+me+and+my+gal+1942': 0.0
        }
    }
    assert get_view_history_stats(history_stats_filepath) == example_dict


#  TO DO: need to come up with test cases for all branches
def test_data_quality_check_0():
    l = "2023-02-05T02:01:32,127552,GET /rate/the+shining+1980=a"
    assert data_quality_check(l) == (False, None)
def test_data_quality_check_1():
    l = "2023-02-05T02:01:32,aaaaaa,GET /rate/the+shining+1980=5"
    assert data_quality_check(l) == (False, None)
def test_data_quality_check_2():
    l = "2023-02-05T02:01:32,127552,GET /rate/the + shining + 1980=5"
    assert data_quality_check(l) == (False, None)


def test_convert_raw_to_dataset():
    rating_stats_filepath = "MLModels/test_ratings.txt"
    lines = open(rating_stats_filepath).readlines()
    data = {"time": [],
            "user": [], 
            "item": [], 
            "rating": []}

    for l in lines:
        try:
            check_res, row = data_quality_check(l)
            if check_res:
                data["time"].append(row[0])
                data["user"].append(row[1])
                data["item"].append(row[2])
                data["rating"].append(row[3])
            else:
                print(
                        f" `{l.strip()}` in {ratings_txt}"   
                    )
        except:
            continue

    df = pd.DataFrame(data)
    df = df.dropna() # remove rows with null values
    # keep both the unique rows and up to the latest three duplicates on user, item, rating columns
    column_names = ["user","item", "rating"]
    unique_rows = df.drop_duplicates(subset=column_names, keep=False)
    duplicate_rows = df[df.duplicated(subset=column_names, keep=False)].groupby(column_names).tail(3)
    df = pd.concat([unique_rows, duplicate_rows])

    refer_dset = Dataset.load_from_df(df[["user", "item", "rating"]], 
                                Reader(rating_scale=(1, 5)))
    
    assert convert_raw_to_dataset(rating_stats_filepath).df.to_csv(path_or_buf=None, columns=["user", "item", "rating"], header=False, index=False) \
            == refer_dset.df.to_csv(path_or_buf=None, columns=["user", "item", "rating"], header=False, index=False)

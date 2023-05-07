import os
import pickle
import random
from time import time

import numpy as np
from scipy.stats import pearsonr
from surprise import SVD, BaselineOnly
from surprise.model_selection import cross_validate

from preprocess import (
    convert_raw_to_dataset,
    data_quality_check
)

SVD_PARAMS = {
    'n_factors': 4,
    'n_epochs': 50,
    'lr_all': 0.005,
    'reg_all': 0.1
}
DATA_PATH = "test/MLModels/fake_ratings.txt"
EVAL_DATA_PATH = "test/MLModels/fake_ratings.txt"

def get_example_model():
    model = SVD(**SVD_PARAMS)
    dataset = convert_raw_to_dataset(DATA_PATH)
    model.fit(dataset.build_full_trainset())

    return model

def convert_bytes(size):
    for unit in ["bytes", "KB", "MB", "GB"]:
        if size < 1024.0:
            # return f"{size:3.2f} {unit}"
            return size
        size /= 1024.0

    # return f"{size:.2f} GB"
    return size

def train_and_eval_accuracy(model_type, data_path = DATA_PATH):
    if model_type == "SVD":
        model = SVD(**SVD_PARAMS)
    elif model_type == "baseline":
        model = BaselineOnly(verbose=False)
    else:
        raise ValueError(f"untested model type: {model_type}")

    # 5-fold cross validation
    start_time = time()
    dataset = convert_raw_to_dataset(DATA_PATH)
    cv = cross_validate(
        model,
        dataset,
        measures=["rmse"],
        n_jobs=5,
    )
    print(
        f"(Measure #1a) {'5-fold valid RMSE':40}:"
        f" {np.mean(cv['test_rmse']):.4f}"
    )

    # fit on full training set
    start_time_tr = time()
    model.fit(dataset.build_full_trainset())

    hit_percentage = calculate_corr_on_test_data(model)

    # print(
    #     f"(Measure #2a) {'cross valid + training time':40}:"
    #     f" {time() - start_time:.2f} sec"
    # )
    # print(
    #     f"(Measure #2b) {'training time':40}:"
    #     f" {time() - start_time_tr:.2f} sec"
    # )

    return model, hit_percentage

def calculate_corr_on_test_data(
    model,
    eval_data_path=EVAL_DATA_PATH
):
    true_ratings = []
    pred_ratings = []

    known_users = model.trainset._raw2inner_id_users.keys()
    known_movies = model.trainset._raw2inner_id_items.keys()

    rating_lines = open(eval_data_path).readlines()
    for l in rating_lines:
        try:
            check_res, row = data_quality_check(l)
            if not check_res:
                continue

            user, item, rating = row[1], row[2], row[3]
        except:
            continue

        if user in known_users and item in known_movies:
            true_ratings.append(float(rating))
            pred_ratings.append(
                model.predict(user, item).est
            )
        

    # print(
    #     f"(Measure #1b) {'Pearson-R w/ test data ratings':40}:"
    #     f" {pearsonr(true_ratings, pred_ratings).statistic:.4f}"
    #     f" (# hits = {len(pred_ratings)} out of {len(rating_lines)})"
    # )
    return pearsonr(true_ratings, pred_ratings).statistic

def measure_inference_speed(model):
    known_users = model.trainset._raw2inner_id_users.keys()
    # sampled_users = random.sample(list(known_users), 100)
    sampled_users = random.sample(list(known_users), 10)

    known_movies = model.trainset._raw2inner_id_items.keys()

    start_time = time()
    preds = [
        [model.predict(u, m) for m in known_movies]
        for u in sampled_users
    ]
    # print(
    #     f"(Measure #3 ) {'inference time (per 100 requests)':40}:"
    #     f" {time() - start_time:.2f} sec"
    # )
    return time() - start_time

def measure_model_size(model):
    tmp_model_path = ".model_eval.tmp"

    pickle.dump(
        model,
        open(tmp_model_path, "wb")
    )

    # print(
    #     f"(Measure #4 ) {'model size':40}:"
    #     f" {convert_bytes(os.path.getsize(tmp_model_path))}"
    # )
    model_size = convert_bytes(os.path.getsize(tmp_model_path))
    os.remove(tmp_model_path)
    return model_size

def evaluate_model(model_type):
    print(f"Evaluating model: {model_type} ... ")
    model, _ = train_and_eval_accuracy(model_type)
    time_ellapsed = measure_inference_speed(model)
    model_size = measure_model_size(model)


if __name__ == "__main__":
    evaluate_model("SVD")
    evaluate_model("baseline")
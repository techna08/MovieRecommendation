'''
Written with help from

https://surprise.readthedocs.io/en/stable/getting_started.html#cv-results-example 
https://surprise.readthedocs.io/en/stable/model_selection.html

SVD hyperparams can be found here: https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD
'''

import random
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from surprise import Dataset, SVD, BaselineOnly
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

from preprocess import convert_raw_to_dataset

svd_param_grid = {
    'n_factors': [4, 10, 100, 500],
    'n_epochs': [5, 20, 50], 
    'lr_all': [0.001, 0.005, 0.02],
    'reg_all': [0.005, 0.02, 0.1]
}

DATA_PATH = "test/MLModels/fake_ratings.txt"
MODEL_OUT_PATH = "test/MLModels/test_model_grid_search.bin"

seed = 2023
random.seed(seed)
np.random.seed(seed)


def grid_search(svd_param_grid=svd_param_grid):
    data = convert_raw_to_dataset(DATA_PATH)

    gs = GridSearchCV(SVD, svd_param_grid, measures=["rmse"], cv=5, n_jobs=8)
    gs.fit(data)

    # best prediction and params
    print(f"SVD (rmse)     : {gs.best_score['rmse']:.4f}")
    print("best setup     :", gs.best_params["rmse"])

    results_df = pd.DataFrame.from_dict(gs.cv_results)
    results_df.to_csv(
        "SVD_grid_search_results_"
        f"{DATA_PATH.replace('/', '_').replace('.txt', '')}.csv"
    )

    baseline = BaselineOnly()
    cv = cross_validate(
        baseline,
        data,
        measures=["rmse"],
        verbose=False
    )
    baseline_rmse = np.mean(cv["test_rmse"])
    print(f"baseline (rmse): {baseline_rmse:.4f}")

    if baseline_rmse < gs.best_score['rmse']:
        print("baseline is chosen, refitting on whole dataset ...")
        baseline.fit(data.build_full_trainset())
        pickle.dump(baseline, open(MODEL_OUT_PATH, "wb"))
    else:
        print("SVD is chosen, refitting on whole dataset ...")
        best_SVD = deepcopy(gs.best_estimator['rmse'])
        best_SVD.fit(data.build_full_trainset())
        pickle.dump(best_SVD, open(MODEL_OUT_PATH, "wb"))
import os
import pickle
import random

from pathlib import Path
import sys
from typing import Union


import numpy as np
from surprise import SVD, BaselineOnly

directory = Path(__file__).resolve().parent.parent
sys.path.append(str(directory))
from provenance import log_prediction


unknown_user_recom = None
known_user_recoms = dict()


def fast_batch_predict(user_id, surprise_model, nbest):
    inner_user_id = surprise_model.trainset.to_inner_uid(user_id)
    item_biases = surprise_model.bi

    if isinstance(surprise_model, SVD):
        item_scores = np.dot(
            surprise_model.qi,
            surprise_model.pu[inner_user_id][None, :].transpose()
        )[:, 0]
        item_scores += item_biases
    elif isinstance(surprise_model, BaselineOnly):
        item_scores = item_biases
    else:
        raise ValueError("unsupported model:", type(surprise_model))

    top_inner_item_ids = np.argsort(item_scores)[::-1][:nbest]

    predictions = [
        surprise_model.trainset.to_raw_iid(it) \
            for it in top_inner_item_ids
    ]

    return predictions


def get_unknown_user_recoms(model):
    all_movie_scores = dict()
    movies = set(model["n_ratings"].keys()).union(
                    set(model["views_t_scores"].keys())
                )

    n_rating_thres = 6
    nbest = 30

    # compute sum of normalized rating and views
    for mv in movies:
        mv_score = 0

        if mv in model["n_ratings"].keys():
            # tune down influence when n_ratings < n_rating_thres
            mv_score += max(1., model["n_ratings"][mv] / n_rating_thres) \
                        * model["rating_t_scores"][mv]

        if mv in model["views_t_scores"].keys():
            mv_score += model["views_t_scores"][mv]

        all_movie_scores[mv] = mv_score

    
    # sort to get n best
    best_movies = [
        pair[0] for pair in \
            sorted(
                all_movie_scores.items(),
                key=lambda p: p[1],
                reverse=True
            )[:nbest]
    ]

    return best_movies


def get_known_user_recoms(user_id, model):
    nbest = 30
    
    # NOTE(Shih-Lun): fast predictions using matrix ops
    predictions = fast_batch_predict(
                        user_id,
                        model["model"],
                        nbest
                    )

    return predictions


def select_results_from_recom(recom_list):
    n_res = random.choice(range(8, 16))
    indices = random.sample(range(len(recom_list)), n_res)

    return [recom_list[i] for i in indices]


def predict(
    user_id: str,
    model: Union[SVD, BaselineOnly],
    if_log = True
) -> str:
    global unknown_user_recom

    try:
        if type(user_id) != str:
            raise TypeError
    except:
        if unknown_user_recom is None:
            unknown_user_recom = get_unknown_user_recoms(model)

        recom = unknown_user_recom

    known_users = model["model"].trainset._raw2inner_id_users.keys()

    if user_id in known_users:
        seen = True
        global known_user_recoms

        if user_id not in known_user_recoms:
            known_user_recoms[user_id] = get_known_user_recoms(user_id, model)

        recom = known_user_recoms[user_id]
    
    else:
        seen = False
        if unknown_user_recom is None:
            unknown_user_recom = get_unknown_user_recoms(model)

        recom = unknown_user_recom

    if if_log:
        log_prediction(
            (user_id, recom, seen), 
            model["model_type"],
            model["training_data_path"],
            model["training_time"],
            model["pipline_version"]
        )

    return ",".join(select_results_from_recom(recom))


if __name__ == "__main__":
    # try unknown user
    res = predict("000")
    print(res)

    # try known users
    model = load_model()
    known_users = model["model"].trainset._raw2inner_id_users.keys()

    for us in random.sample(list(known_users), 20):
        res = predict(us)
        print(res)




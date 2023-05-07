import pickle
import logging
import datetime

from surprise import BaselineOnly, SVD
import sys
import os

from pathlib import Path

directory = Path(__file__).resolve().parent.parent
sys.path.append(str(directory))
from provenance import get_pipeline_version


sys.path.append(os.path.dirname(__file__))
from preprocess import (
    convert_raw_to_dataset,
    get_view_history_stats,
    get_rating_stats
)


logging.basicConfig(
    format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SVD_PARAMS = {
    'n_factors': 4,
    'n_epochs': 50,
    'lr_all': 0.005,
    'reg_all': 0.1
}

model_type = "baseline"
ratings_path = "data_0215_900k/ratings_train.txt"
views_path = "data_0215_900k/mpg.txt"


def train_and_dump_model(
    model_type,
    train_data_path,
    view_history_path,
    model_out_path="model.bin"
):
    if model_type == "SVD":
        model = SVD(**SVD_PARAMS)
    elif model_type == "baseline":
        model = BaselineOnly(verbose=False)
    else:
        raise ValueError(f"untested model type: {model_type}")

    dataset = convert_raw_to_dataset(train_data_path)
    model.fit(dataset.build_full_trainset())
    logger.info("model training completed")

    


    # to handle users w/o ratings
    views_stats = get_view_history_stats(view_history_path)
    ratings_stats = get_rating_stats(train_data_path)

    pickle.dump(
        {
            "model": model,
            "model_type": str(type(model)),
            "pipline_version": get_pipeline_version(),
            "training_data_path": str(train_data_path),
            "training_time": datetime.datetime.now(),
            **views_stats,
            **ratings_stats,
        },
        open(model_out_path, "wb")
    )

    return


if __name__ == "__main__":
    train_and_dump_model(
        "baseline",
        ratings_path,
        views_path,
        model_out_path="model_baseline.bin"
    )
    # train_and_dump_model(
    #     "baseline",
    #     ratings_path,
    #     views_path,
    #     model_out_path = "baseline_model.bin"
    # )

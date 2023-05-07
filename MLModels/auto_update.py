import os
import sys
import datetime
import random
import pickle
import json
from typing import Dict

sys.path.append("MLModels/")
sys.path.append(os.path.dirname(__file__) + "/../")
from train import train_and_dump_model
from evaluate import calculate_corr_on_test_data
from flask_app import load_model

TRAIN_LOGS_ROOT = sys.argv[1]
DATA_OUT_DIR = sys.argv[2]
MODEL_OUT_DIR = sys.argv[3]
MODEL_TYPE = sys.argv[4]
USE_CACHED_DATA = sys.argv[5] == "True"

data_files = None

# NOTE(Shih-Lun): can skip data collection
if USE_CACHED_DATA and len(sys.argv) == 9:
    data_files = {
        "ratings_train_set": sys.argv[6],
        "ratings_test_set": sys.argv[7],
        "view_hist_all": sys.argv[8]
    }
elif USE_CACHED_DATA:
    data_files = json.load(
        open(
            os.path.join(DATA_OUT_DIR, "cached_data_path.json")
        )
    )
    print("[info] using cached data:")
    print(json.dumps(data_files, indent=4))


def get_three_day_ago_date() -> datetime.date:
    """Returns the date 3 days ago"""
    time = datetime.datetime.now() - datetime.timedelta(hours=72)

    return time.date()


def collect_all_logs_and_dump(
    start_date: datetime.date
) -> Dict[str, os.PathLike]:
    """Collects ratings and view history log lines from `start_date` to now
    
    Collates individual log files and
    Combines them into three files (dumped to .txt):
        (ratings_train_set, ratings_test_set, view_hist_all),
    Returns the paths of dumped .txt's in a dict, e.g.,
        {
            "ratings_train_set": "data_out/ratings_train_2023-04-10T1108.txt",
            "ratings_test_set": "data_out/ratings_test_2023-04-10T1108.txt",
            "view_hist_all": "data_out/view_history_2023-04-10T1108.txt"
        }
    """

    views_lines = []
    ratings_lines = []

    for d in range(4):
        check_date = start_date + datetime.timedelta(days=d)
        check_date_dir = str(check_date)
        check_date_dir = os.path.join(TRAIN_LOGS_ROOT, check_date_dir)

        if not os.path.exists(check_date_dir):
            continue

        for h in range(24):
            st_hour = \
                datetime.datetime(
                    check_date.year,
                    check_date.month,
                    check_date.day
                ) + datetime.timedelta(hours=h)
            ed_hour = \
                datetime.datetime(
                    check_date.year,
                    check_date.month,
                    check_date.day
                ) + datetime.timedelta(hours=h+1)

            if st_hour > datetime.datetime.now():
                break

            check_hour_dir = os.path.join(
                check_date_dir,
                str(st_hour.time())[:2] + "00-" + \
                str(ed_hour.time())[:2] + "00"
            )

            if not os.path.exists(check_hour_dir):
                continue
            
            try:
                _hour_ratings = open(
                    os.path.join(check_hour_dir, "ratings.txt")
                ).readlines()
            except:
                _hour_ratings = []

            try:
                _hour_views = open(
                    os.path.join(check_hour_dir, "mpg.txt")
                ).readlines()
            except:
                _hour_views = []

            print(
                "[info] "
                f"collecting from: {st_hour}, "
                f"got {len(_hour_ratings)} ratings & "
                f"{len(_hour_views)} views"
            )

            # NOTE(Shih-Lun): subsample view history 
            #                 to prevent excessive dump size
            if len(_hour_views):
                _hour_views = random.sample(
                                _hour_views,
                                int(0.03 * len(_hour_views))
                            )
            print(f"[info] subsample to {len(_hour_views)} views")

            views_lines.extend(_hour_views)
            ratings_lines.extend(_hour_ratings)


    print(
        f"[info] finished data collection, "
        f"got {len(ratings_lines)} ratings & {len(views_lines)} views"
    )
    data_time_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H%M")


    # NOTE(Shih-Lun): train-test splits for ratings
    ratings_train, ratings_test = (
        ratings_lines[:int(0.8 * len(ratings_lines))],
        ratings_lines[int(0.8 * len(ratings_lines)):]
    )

    ratings_train_file = \
        os.path.join(DATA_OUT_DIR, f"ratings_train_{data_time_tag}.txt")

    ratings_test_file = \
        os.path.join(DATA_OUT_DIR, f"ratings_test_{data_time_tag}.txt")

    views_file = \
        os.path.join(DATA_OUT_DIR, f"view_history_{data_time_tag}.txt")
    

    with open(ratings_train_file, "w") as f:
        f.writelines(ratings_train)

    with open(ratings_test_file, "w") as f:
        f.writelines(ratings_test)

    with open(views_file, "w") as f:
        f.writelines(views_lines)

    
    return {
        "ratings_train_set": ratings_train_file,
        "ratings_test_set": ratings_test_file,
        "view_hist_all": views_file
    }


if __name__ == "__main__":
    if data_files is None:
        if not os.path.exists(DATA_OUT_DIR):
            os.makedirs(DATA_OUT_DIR)
        
        start_date = get_three_day_ago_date()

        data_files = collect_all_logs_and_dump(start_date)
        print("[info] wrote dataset to:")
        print(json.dumps(data_files, indent=4))

        with open(
            os.path.join(DATA_OUT_DIR, "cached_data_path.json"),
            "w"
        ) as f:
            f.write(json.dumps(data_files, indent=4))
            f.write("\n")


    if not os.path.exists(MODEL_OUT_DIR):
        os.makedirs(MODEL_OUT_DIR)

    model_time_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H%M")
    model_out_path = os.path.join(
        MODEL_OUT_DIR,
        f"model_{MODEL_TYPE}.bin"
    )

    train_and_dump_model(
        MODEL_TYPE,
        data_files["ratings_train_set"],
        data_files["view_hist_all"],
        model_out_path=model_out_path,
    )

    test_corr = calculate_corr_on_test_data(
        load_model(model_path=model_out_path)["model"],
        data_files["ratings_test_set"]
    )

    print(
        f"[info] trained {MODEL_TYPE} model, "
        f"test corr = {test_corr:.4f}"
    )

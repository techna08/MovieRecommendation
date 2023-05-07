import os
import sys
import datetime
from collections import defaultdict
from typing import List, Union
sys.path.append(os.path.dirname(__file__) + "/../MLModels/")

import numpy as np

from preprocess import data_quality_check

# NOTE(Shih-Lun): remember to set env var
KAFKA_LOGS_ROOT = os.environ["KAFKA_LOGS_ROOT"]


def read_ratings_file(
        ratings_txt: Union[os.PathLike, str],
        verbose: bool = False
    ) -> List[int]:
    """Reads .txt file containing per-line rating entries like

    `2023-02-07T10:08:56,01234567,GET /rate/the+terminator+1984=4`,
    parses out the rating (1 ~ 5), and returns all valid ratings
    """
    print("[info] reading ratings file:", ratings_txt)

    ratings = []

    try:
        entries = open(ratings_txt).readlines()
    except FileNotFoundError:
        return []
    
    seen_ratings = defaultdict(int)

    for l in entries:
        try:
            check_res, row = data_quality_check(l)
            if check_res:
                user = row[1]
                item = row[2]
                rating = row[3]
                
                seen_ratings[(user, item, rating)] += 1
                # NOTE(Shih-Lun):
                # too many duplicated ratings --> suspicious case, skipped
                if seen_ratings[(user, item, rating)] > 3:
                    continue
            else: # invalid format
                continue
        except: # invalid format (parsing error)
            if verbose:
                print(
                    f"[WARNING] got invalid rating entry `{l.strip()}` "
                    f"in {ratings_txt}"
                )
            continue

        ratings.append(rating)

    return ratings


def count_view_hours(
        views_txt: Union[os.PathLike, str],
        skip_check: bool = True,
    ) -> float:
    """Reads .txt file containing per-line view history entries like

    `2023-03-16T01:00,01234567,GET /data/m/requiem+2006/31.mpg`,
    counts the # of lines (i.e., 1 minute movie views),
    and returns # of movie hours watched
    """
    print("[info] reading view history file:", views_txt)

    try:
        entries = open(views_txt).readlines()
    except FileNotFoundError:
        return 0.
    
    n_views = 0
    if not skip_check:
        for l in entries:
            try:
                check_res, row = data_quality_check(l, type = "mpg")
                if check_res:
                    n_views += 1
                else: # invalid format
                    continue
            except: # invalid format (parsing error)
                continue
    else:
        n_views = len(entries)

    return n_views / 60


def get_week_ago_time() -> datetime.date:
    """Returns the time six days ago"""
    time = datetime.datetime.now() - datetime.timedelta(days=6)
    
    return time


def get_day_ago_time() -> datetime.datetime:
    """Returns the time 24 hours ago"""
    time = datetime.datetime.now() - datetime.timedelta(hours=24)

    return time


def convert_to_midnight(time: datetime.datetime) -> datetime.datetime:
    return datetime.datetime.combine(
                time.date(),
                datetime.time.min
            )


def compute_online_metrics(
    get_one_day_only=False
):
    """Hourly (~24 hours ago) and daily (~7 days ago)

    i) total hours of movie views
    ii) mean ratings
    """
    daily_totals = defaultdict(float)
    hourly_totals = defaultdict(float)

    daily_ratings = dict()
    hourly_ratings = dict()
 
    if not get_one_day_only:
        furthest_ago_time = get_week_ago_time()
        furthest_ago_time = convert_to_midnight(furthest_ago_time)
    else:
        furthest_ago_time = get_day_ago_time()
        furthest_ago_time = convert_to_midnight(furthest_ago_time)

    day_ago_time = get_day_ago_time()
    now_time = datetime.datetime.now()

    n_days = 7 if not get_one_day_only else 2
    n_last_24hr_views = 0
    n_last_24hr_ratings = 0
    last_24hr_rating_accum = 0
    last_hr_rating = None

    for d in range(n_days):
        check_date = furthest_ago_time + datetime.timedelta(days=d)

        check_date_dir = str(check_date.date())
        check_date_dir = os.path.join(KAFKA_LOGS_ROOT, check_date_dir)
        
        if not os.path.exists(check_date_dir):
            continue

        _day_ratings_record = ([], [])

        for h in range(24):
            st_hour = check_date + datetime.timedelta(hours=h)
            ed_hour = check_date + datetime.timedelta(hours=h+1)

            if st_hour > now_time:
                break

            check_hour_dir = os.path.join(
                check_date_dir,
                str(st_hour.time())[:2] + "00-" + str(ed_hour.time())[:2] + "00"
            )

            if not os.path.exists(check_hour_dir):
                continue

            _hour_views = count_view_hours(
                os.path.join(check_hour_dir, "mpg.txt")
            )
            _hour_ratings = read_ratings_file(
                os.path.join(check_hour_dir, "ratings.txt")
            )

            daily_totals[str(check_date.date())] += _hour_views
            
            if len(_hour_ratings):
                _day_ratings_record[0].append(np.average(_hour_ratings))
                _day_ratings_record[1].append(len(_hour_ratings))

            if len(_hour_ratings) > 500:
                last_hr_rating = np.average(_hour_ratings)

            if st_hour > day_ago_time:
                hour_key_str = \
                    str(check_date.date()) + "T" + \
                    str(st_hour.time())[:2] + "-" + str(ed_hour.time())[:2]
                hourly_totals[hour_key_str] = _hour_views
 
                if ed_hour < now_time:
                    n_last_24hr_views += _hour_views


                if len(_hour_ratings):
                    hourly_ratings[hour_key_str] = {
                        "mean_rating": np.average(_hour_ratings),
                        "std": np.std(_hour_ratings)
                    }

                    last_24hr_rating_accum += \
                        hourly_ratings[hour_key_str]["mean_rating"] * len(_hour_ratings)
                    n_last_24hr_ratings += len(_hour_ratings)


        if len(_day_ratings_record[0]):
            daily_rating_avg = np.average(
                                _day_ratings_record[0], 
                                weights=_day_ratings_record[1]
                            ) 
            daily_ratings[str(check_date.date())] = {
                "mean_rating": np.average(
                                _day_ratings_record[0], 
                                weights=_day_ratings_record[1]
                            ),
                "hourly_std": np.average(
                        (np.array(_day_ratings_record[0]) - daily_rating_avg) ** 2,
                        weights=_day_ratings_record[1]
                    )
            }


        last_24hr_rating = last_24hr_rating_accum / n_last_24hr_ratings


    return {
        "rating": {
            "hourly": hourly_ratings,
            "daily": daily_ratings
        },
        "view_hours": {
            "hourly": dict(hourly_totals),
            "daily": dict(daily_totals)
        },
        "last_24hr_view_hours": n_last_24hr_views,
        "last_24hr_rating": last_24hr_rating,
        "last_hr_rating": last_hr_rating
    }


if __name__ == "__main__":
    import json
    # NOTE(Shih-Lun): main() for expedient testing
    _week_ago_time = get_week_ago_time()
    print("a week ago:", _week_ago_time)

    _day_ago_time = get_day_ago_time()
    print("a day ago:", _day_ago_time)

    _metrics = compute_online_metrics(get_one_day_only=True)

    print("metrics:", json.dumps(_metrics, indent=4))

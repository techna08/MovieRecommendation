import random
import time
import datetime
import sys
import os
sys.path.append(os.path.dirname(__file__) + "/../../")

import pytest
from pytest_mock import mocker

import online_eval
from online_eval.online_evaluate import *

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

@pytest.fixture
def tmp_log_folders():
    fake_dates = [
        "2023-03-13",
        "2023-03-14",
        "2023-03-15",
        "2023-03-16",
        "2023-03-17",
        "2023-03-18",
        "2023-03-19",
        "2023-03-20"
    ]
    fake_hours = ["0000-0100", "0100-0200"]

    for date in fake_dates:
        for hour in fake_hours:
            os.makedirs(os.path.join(date, hour))

    yield

    for date in fake_dates:
        for hour in fake_hours:
            os.rmdir(os.path.join(date, hour))
        os.rmdir(date)

def random_time():
    d = random.randint(
        int(time.time()) - 1000000, 
        int(time.time()) + 1000000
    )

    return datetime.datetime.fromtimestamp(d)


def test_convert_to_midnight():
    for i in range(5):
        _time = random_time()
        _midnight = datetime.datetime(
            year=_time.year,
            month=_time.month,
            day=_time.day
        )
        
        assert _midnight == convert_to_midnight(_time)


def test_get_day_ago_time():
    day_ago_time = get_day_ago_time()
    now_time = datetime.datetime.now()
    assert isinstance(day_ago_time, datetime.datetime)


    delta = now_time - day_ago_time
    assert delta.days == 1
    assert delta.microseconds < 1000


def test_get_week_ago_time():
    week_ago_time = get_week_ago_time()
    now_time = datetime.datetime.now()
    assert isinstance(week_ago_time, datetime.datetime)


    delta = now_time - week_ago_time
    assert delta.days == 6
    assert delta.microseconds < 1000


def test_read_ratings_file(tmp_ratings_file):
    ratings = read_ratings_file(tmp_ratings_file)
    assert len(ratings) == len(TEST_RATINGS)


def test_count_view_hours(tmp_views_file):
    hours = count_view_hours(tmp_views_file)
    assert hours == len(TEST_VIEWS) / 60


def test_compute_online_metrics(mocker, tmp_log_folders):
    assert os.path.exists(KAFKA_LOGS_ROOT)

    mocker.patch(
        "online_eval.online_evaluate.get_week_ago_time",
        return_value=datetime.datetime(2023, 3, 13, 16, 0, 0)
    )
    mocker.patch(
        "online_eval.online_evaluate.get_day_ago_time",
        return_value=datetime.datetime(2023, 3, 19, 16, 0, 0)
    )
    mocker.patch(
        "online_eval.online_evaluate.count_view_hours",
        return_value=183.25
    )
    mocker.patch(
        "online_eval.online_evaluate.read_ratings_file",
        return_value=[3, 2, 3, 5, 4, 4]
    )

    metrics = compute_online_metrics()
    
    assert isinstance(metrics, dict)
    assert "rating" in metrics and "view_hours" in metrics
    for mtr in ["rating", "view_hours"]:
        assert "hourly" in metrics[mtr]
        assert "daily" in metrics[mtr]

import os
import logging
from collections import defaultdict

import numpy as np
from surprise import Dataset, Reader
import pandas as pd
import re
import requests

logging.basicConfig(
    format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

movie_server_ip = os.getenv("MOVIE_SERVER_IP")
print("[INFO] movie_server_ip =", movie_server_ip)

valid_movies = set()
invalid_movies = set()

# NOTE(Shih-Lun): discovered from Kafka stream
invalid_movie_names = [
    "the+shawshaink+redemption+1994", 
    "machuca+200q4", 
    "the+godfather+1c972", 
    "noisees+off...+1992", 
    "toy+story+j1995", 
    "life+is+beautiful+19997", 
    "stroszek+H1977", 
    "pulp+fictionC+1994", 
    "theO+usual+suspects+1995", 
    "one+flew+over+the+cuckoos+nest+1r975", 
    "adam+20090", 
    "spirited+aJway+2001", 
    "goodfelLlas+1990", 
    "schindlers+llist+1993", 
    "the+matrix+1e999", 
    "the+sacrifice+19B86", 
    "the+bestT+of+youth+2003", 
    "the+dark+knight0+2008", 
    "the+good_+the+bad+and+tqhe+ugly+1966", 
    "the+usual+msuspects+1995", 
    "the+imitatibon+game+2014", 
    "raiderfs+of+the+lost+ark+1981", 
    "star+wars+episode+iii+-+revenge+of+the+sith+200M5", 
    "pulp+fictigon+1994", 
    "land+andM+freedom+1995", 
    "finding+nezmo+2003", 
    "the+secret+world+of+arJrietty+2010", 
    "the+shawshank+redemptxion+1994", 
    "big+shotE+confessions+of+a+campus+bookie+2002", 
    "Ogladiator+2000", 
    "viva+zapata+1952t", 
    "one+flew+Fover+the+cuckoos+nest+1975", 
    "the+shawshank+redemption+19924", 
    "the+lord+of+the+rin7gs+the+return+of+the+king+2003", 
    "spirMited+away+2001", 
    "qpulp+fiction+1994", 
    "the+bourne+iden5tity+2002", 
    "the+godfather+19762", 
    "seven+jsamurai+1954", 
    "the+shawshaXnk+redemption+1994", 
    "terminaytor+2+judgment+day+1991", 
    "indiana+jones+and+the+lasRt+crusade+1989", 
    "8the+imitation+game+2014", 
    "casabKlanca+1942", 
    "schindlers+li6st+1993", 
    "theR+silence+of+the+lambs+1991", 
    "the+sAhawshank+redemption+1994", 
    "harry7+potter+and+the+prisoner+of+azkaban+2004", 
    "alicez+1988", 
    "the+secrmet+in+their+eyes+2009", 
    "the+avengers+201U2", 
    "willy+wonka++the+chocolate+factorYy+1971", 
    "one+flew+oqver+the+cuckoos+nest+1975", 
    "one+flewM+over+the+cuckoos+nest+1975", 
    "the+detectmive+2+2011", 
    "the+shawshanDk+redemption+1994", 
    "the+staircase+20N04", 
    "the+decline+of+western+civilization+parGt+ii+the+metal+years+1988", 
    "tfhe+shawshank+redemption+1994", 
    "my+nZeighbor+totoro+1988", 
    "monty+pYython+and+the+holy+grail+1975", 
    "the+shawshank+redempt3ion+1994", 
    "bZlackbeard_+the+pirate+1952", 
    "trhe+lady+eve+1941", 
    "raiders+of+the+Zlost+ark+1981", 
    "swimmingM+to+cambodia+1987", 
    "the+dark+zknight+2008", 
    "jiro+dreamxs+of+sushi+2011", 
    "the+human+condition+i+no+greater+love+1959x", 
    "my+neighboNr+totoro+1988", 
    "whisper+of+the+heart+19q95", 
    "pirwates+of+the+caribbean+the+curse+of+the+black+pearl+2003", 
    "12+angry+men3+1957", 
    "sePven+samurai+1954", 
    "pulp+fictioGn+1994", 
    "se7enz+1995", 
    "harry+potter+and+the+chamber+of+secrets+20D02", 
    "3harry+potter+and+the+deathly+hallows+part+2+2011", 
    "the+usual+suspwects+1995", 
    "the0+godfather+1972", 
    "raiders+of+the+lost+ark+1G981", 
    "the+shawshank+redemption3+1994", 
    "raiderts+of+the+lost+ark+1981", 
    "pulp+fiqction+1994", 
    "the+Qgodfather+1972", 
    "lornas+silence+200V8", 
    "the+inherfitance+2003", 
    "the+good_+the+bad+and+the+ugly+19616", 
    "the+lord+of+the+rings+the+fellowship+of+the+ringG+2001", 
    "pirates+of+the+caribbean+the+curse+oSf+the+black+pearl+2003", 
    "the+sorrow+and+the+pity+g1969", 
    "th2e+bleeding+house+2011", 
    "heat+1n995", 
    "the+shawshank+redempQtion+1994"
]
invalid_movie_names = set(invalid_movie_names)

def get_rating_stats(ratings_txt):
    lines = open(ratings_txt).readlines()
    data = { "time": [],
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
                # print(
                #         f" `{l.strip()}` in {ratings_txt}"   
                #     )
                continue
        except:
            # print(f"[WARNING] Invalid entry {l.strip()} in {ratings_txt}")
            continue

    df = pd.DataFrame(data)
    df = df.dropna() # remove rows with null values
    # keep both the unique rows and up to the latest three duplicates on user, item, rating columns
    column_names = ["user","item", "rating"]
    unique_rows = df.drop_duplicates(subset=column_names, keep=False)
    duplicate_rows = df[df.duplicated(subset=column_names, keep=False)].groupby(column_names).tail(3)
    df = pd.concat([unique_rows, duplicate_rows])

    n_ratings = df["item"].value_counts().to_dict()
    df["rating"] = df["rating"].astype(float)
    all_movie_ratings = df.groupby("item")["rating"].mean().to_dict()

    mean_rating = np.mean(list(all_movie_ratings.values()))
    std_rating = np.std(list(all_movie_ratings.values()))

    # normalize ratings
    for k in all_movie_ratings.keys():
        all_movie_ratings[k] = \
            (all_movie_ratings[k] - mean_rating) / std_rating

    return {
        "n_ratings": n_ratings,
        "rating_t_scores": all_movie_ratings
    }

def get_view_history_stats(hist_txt):
    all_movie_views = defaultdict(int)
    data = {"time": [],
            "user": [], 
            "item": [], 
            "minute": []}

    for l in open(hist_txt).readlines():
        try:
            check_res, row = data_quality_check(l, type = "mpg")
            if check_res:
                data["time"].append(row[0])
                data["user"].append(row[1])
                data["item"].append(row[2])
                data["minute"].append(row[3])
            else:
                # print(
                #         f" `{l.strip()}` in {hist_txt}"   
                #     )
                continue
        except:
            # print(f"[WARNING] Invalid entry {l.strip()} in {hist_txt}")
            continue
    
    df = pd.DataFrame(data)
    df = df.dropna() # remove rows with null values
    df = df.drop_duplicates() # remove duplicates

    df_groupby = df.groupby("item") \
                   .filter(lambda x: len(x) > 3) \
                   .groupby("item") \
                   .count()["user"]

    mean_views = df_groupby.mean()
    std_views = df_groupby.std()

    # normalize number of views
    all_movie_views = ((df_groupby - mean_views) / (std_views + 1e-5)).to_dict()
    return {
        "views_t_scores": all_movie_views
    }

def data_quality_check(
    line,
    type="rating",
    verbose=False,
):
    time, user, item, rating, minute = None, None, None, None, None
    if type == "rating":
        time, user = line.split(",")[0].strip(), line.split(",")[1].strip()
        item = line.split("/")[-1].split("=")[0].strip() 
        rating = line.split("/")[-1].split("=")[1].strip()
    else:
        time, user = line.split(",")[0].strip(), line.split(",")[1].strip()
        item, minute = line.split("/")[-2].strip(), line.split("/")[-1].strip()

    # Check datatype of ratings
    if type == "rating" and (not rating.isdigit() or \
       int(rating) < 1 or int(rating) > 5):
        if verbose:
            print("[WARNING] Invalid rating entry")
        return False, None

    # Check datatype of users
    if not user.isdigit():
        if verbose:
            print("[WARNING] Invalid user entry")
        return False, None

    # Check format of movies
    replaced_item = item.replace(" ","")
    
    if not re.match(r"^\+*[^+]+(\++[^+]+)*\+?$", item) or \
       len(replaced_item) != len(item):
        if verbose:
            print("[WARNING] Invalid item entry")
        return False, None

    # Check special invalid movie name cases
    if item in invalid_movie_names:
        return False, None

    year_str = item.split("+")[-1]
    movie_str = "+".join(item.split("+")[:-1])

    # Additional rule -- year string
    if len(year_str) != 4 or not year_str.isdigit():
        return False, None
    # Additional rule -- invalid movie name, containing upper case
    if movie_str.lower() != movie_str:
        return False, None
    if not movie_str.replace("+", "").isalnum():
        return False, None

    if item in valid_movies:
        pass
    elif item in invalid_movies:
        return False, None
    elif movie_server_ip is not None:
        # request
        response = requests.get(
            f"http://{movie_server_ip}:8080/movie/{item}"
        ).json()
        
        if "message" in response and response["message"] == "movie not found":
            invalid_movies.add(item)
            print("[INFO] got invalid movie:", item)
            return False, None
        else:
            valid_movies.add(item)

    if type == "rating":
        return True, (time, user, item, int(rating))
    else:
        return True, (time, user, item, minute)

def convert_raw_to_dataset(
    ratings_txt,
    verbose=False,
):
    lines = open(ratings_txt).readlines()
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
                if verbose:
                    print(
                            f" `{l.strip()}` in {ratings_txt}"   
                        )
        except:
            if verbose:
                print(
                    f"[WARNING] Invalid unkown entry `{l.strip()}` "
                    f"in {ratings_txt}"
                )
            continue

    # print(len(data),data['rating'])
    df = pd.DataFrame(data)
    df = df.dropna() # remove rows with null values
    # keep both the unique rows and up to the latest three duplicates on user, item, rating columns
    column_names = ["user","item", "rating"]
    unique_rows = df.drop_duplicates(subset=column_names, keep=False)
    duplicate_rows = df[df.duplicated(subset=column_names, keep=False)].groupby(column_names).tail(3)
    df = pd.concat([unique_rows, duplicate_rows])

    dset = Dataset.load_from_df(df[["user", "item", "rating"]], 
                                Reader(rating_scale=(1, 5)))

    # print(type(dset))
    logger.info(
        f"got {len(set(df['user']))} users, {len(set(df['item']))} items, "
        f"{df.shape[0]} ratings, sparsity = "
        f"{100 - 100 * df.shape[0] / len(set(df['item'])) / len(set(df['user'])):.2f} %"
        f"\nThere are {df.shape[0]} quality data out of {len(lines)}."
    )

    return dset

if __name__ == "__main__":
    dset = convert_raw_to_dataset("MLModels/test_ratings.txt")
    rate_stats = get_rating_stats("MLModels/test_ratings.txt")

    for k in list(rate_stats["n_ratings"].keys())[:10]:
        print(
            k,
            rate_stats["n_ratings"][k],
            rate_stats["rating_t_scores"][k]
        )
    hist_stats = get_view_history_stats("MLModels/test_mpg.txt")
    for k in list(hist_stats["views_t_scores"].keys())[:10]:
        print(
            k,
            hist_stats["views_t_scores"][k]
        )
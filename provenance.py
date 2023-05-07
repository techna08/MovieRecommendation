from collections import defaultdict
import datetime
import json
import os
import pandas as pd
import subprocess

DATA_PATH = 'var'

def log_prediction(
    prediction,
    model_type,
    training_data_path,
    training_timestamp,
    pipeline_version,
):
    '''
    The function creates a log file path based on the date extracted from the model_version, 
    and writes the DataFrame (including training and predicting users) to the log file.

    Args:
    - dataset(tuple): a tuple containing user information, item recommendations, and seen items.
    - model_type(str): a string indicating the type of model.
    - model_version(str): an optional string indicating the version of the model.

    Returns:
    - cur_model_version (str): A string indicating the current model version for the given model type.
    '''    
    user_id, recom, seen = prediction
    df = pd.DataFrame({
            "time": datetime.datetime.now(),
            "user_id": [user_id],
            "is_seen_user": [seen],
            "recommendations": [recom],
            "model_type": [str(model_type)],
            "training_data": [str(training_data_path)],
            "training_timestamp": [str(training_timestamp)],
            "pipeline_version": [str(pipeline_version)],
        })

    today = str(datetime.date.today())
    log_path = f'{DATA_PATH}/{today}.csv'
    if os.path.exists(log_path):
        df.to_csv(log_path, index=False, mode='a', header=False)
    else:
        print("*******",log_path)
        df.to_csv(log_path, index=False)


def get_pipeline_version() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()



if __name__ == "__main__":

    # print(get_pipeline_version())
    from MLModels.predict import load_model, predict
    import random
    model = load_model(model_path = "model_svd.bin")
    known_users = model["model"].trainset._raw2inner_id_users.keys()
    for us in random.sample(list(known_users), 20):
        res = predict(us, "model_svd.bin")
        print(res)




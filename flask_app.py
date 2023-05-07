import os, pickle, random 
from flask import Flask, request, jsonify 
from MLModels import predict
import re
import sys

app = Flask('movie_recommend_service')
model_path = sys.argv[1]
model = None

# create log file root directry if not exists
logFile_root = "var"
if not os.path.exists(logFile_root):
    os.makedirs(logFile_root)

def load_model(model_path=None):
    if model_path is None:
        model_path = os.getenv("RECOM_MODEL_PATH", default='MLModels/model.bin')
    
    return pickle.load(
                open(
                    model_path,
                    "rb"
                )
            )


@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    global model

    res = predict.predict(
            userid,
            model
        )
    # uncomment this to change the format
    #res=res.replace('+',' ')
    #res=re.sub(r'.{5},', ',', res)
    #res = res[:-5]
    
    return res


if __name__ == '__main__':
    model = load_model(
        model_path=model_path
    )
    app.run(port=8081, host='0.0.0.0', debug=False)
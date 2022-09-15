import json
import joblib
import numpy as np
import os

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'movielens_sar_model.pkl')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    # data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    # predictions = model.predict(data)
    top_k = model.recommend_k_items(raw_data, top_k=10, remove_seen=True)
    
    # Return the predictions as JSON
    return json.dumps(top_k)

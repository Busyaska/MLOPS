import pandas as pd
import joblib
from os.path import join
from datetime import datetime

def test_model():
    df = pd.read_csv('dataset/cleared_data.csv', index_col=0)
    X = df.drop('Anxiety_Level_(1-10)', axis=1).sample(1).values

    with open('best_model_path.txt', 'r') as path_file:
        path = join(path_file.readline().strip(), 'model.pkl')
        model = joblib.load(path)
    
    prediction = model.predict(X)
    
    with open('model_log.txt', 'a+') as log_file:
        log_file.write(f'{datetime.now()} - Prediction: {prediction}\n')

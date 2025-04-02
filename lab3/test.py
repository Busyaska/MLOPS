import requests
import json
import pandas as pd


df = pd.read_csv('dataset/cleared_data.csv')
X = df.drop('Anxiety_Level_(1-10)', axis=1).sample(1)

url = "http://127.0.0.1:5003/invocations" 
headers = {"Content-Type": "application/json"}

data = {'dataframe_split': X.to_dict(orient="split")}

response = requests.post(url, headers=headers, data=json.dumps(data))

print("Response:", response.json())
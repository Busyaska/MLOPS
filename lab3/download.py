import kagglehub
import shutil
import os
import pandas as pd


def download():
    destination = "dataset"
    os.makedirs(destination, exist_ok=True)
    path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset", force_download=True)

    for file_name in os.listdir(path):
        shutil.move(os.path.join(path, file_name), os.path.join(destination, file_name))


def preprocess():
    df = pd.read_csv('dataset/enhanced_anxiety_dataset.csv')

    df.rename(lambda x: x if ' ' not in x else x.replace(' ', '_'), axis='columns', inplace=True)

    numeric_columns = df.select_dtypes('number').columns

    for column in numeric_columns:
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = (df[column] - min_value) / (max_value - min_value)

    df['Smoking'] = pd.get_dummies(df.Smoking).astype('Int64')['Yes']
    df['Family_History_of_Anxiety'] = pd.get_dummies(df.Family_History_of_Anxiety).astype('Int64')['Yes']
    df['Dizziness'] = pd.get_dummies(df.Dizziness).astype('Int64')['Yes']
    df['Medication'] = pd.get_dummies(df.Medication).astype('Int64')['Yes']
    df['Recent_Major_Life_Event'] = pd.get_dummies(df.Recent_Major_Life_Event).astype('Int64')['Yes']

    one_hot_occupation = pd.get_dummies(df['Occupation'], dtype='Int64')
    df = pd.concat([df, one_hot_occupation], axis=1)
    df.drop('Occupation', axis=1, inplace=True)

    one_hot_gender = pd.get_dummies(df['Gender'], dtype='Int64')
    df = pd.concat([df, one_hot_gender], axis=1)
    df.drop('Gender', axis=1, inplace=True)

    df.to_csv('dataset/cleared_data.csv')


download()
preprocess()

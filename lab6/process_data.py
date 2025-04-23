import joblib
import pandas as pd


def process_user_data(user_data: dict[str: str|int]) -> pd.DataFrame:
    expected_data = {
            "Smoking": 0,
            "Family_History_of_Anxiety": 0,
            "Dizziness": 0,
            "Medication": 0,
            "Recent_Major_Life_Event": 0,
            "Age": 0,
            "Sleep_Hours": 0,
            "Physical_Activity_(hrs/week)": 0,
            "Caffeine_Intake_(mg/day)": 0,
            "Alcohol_Consumption_(drinks/week)": 0,
            "Stress_Level_(1-10)": 0,
            "Heart_Rate_(bpm)": 0,
            "Breathing_Rate_(breaths/min)": 0,
            "Sweating_Level_(1-5)": 0,
            "Therapy_Sessions_(per_month)": 0,
            "Diet_Quality_(1-10)": 0,
            "Occupation_Artist": 0,
            "Occupation_Athlete": 0,
            "Occupation_Chef": 0,
            "Occupation_Doctor": 0,
            "Occupation_Engineer": 0,
            "Occupation_Freelancer": 0,
            "Occupation_Lawyer": 0,
            "Occupation_Musician": 0,
            "Occupation_Nurse": 0,
            "Occupation_Other": 0,
            "Occupation_Scientist": 0,
            "Occupation_Student": 0,
            "Occupation_Teacher": 0,
            "Gender_Female": 0,
            "Gender_Male": 0,
            "Gender_Other": 0
    }

    scaler = joblib.load('models/scaler.pkl')

    user_df = pd.DataFrame([user_data])
    expected_df = pd.DataFrame([expected_data])

    user_df.rename(lambda x: x if ' ' not in x else x.replace(' ', '_'), axis='columns', inplace=True)

    numeric_columns = user_df.select_dtypes('number').columns
    object_columns = user_df.select_dtypes('object').columns

    expected_df[numeric_columns] = scaler.transform(user_df[numeric_columns])

    for column in object_columns:
        value = user_df[column].values[0]
        if value in ['Yes', 'No']:
            expected_df[column] = int(value == 'Yes')
        else:
            expected_df[f'{column}_{value}'] = 1

    return expected_df

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_heart_stroke_data(filename):
    """
    Loads the Stroke Prediction Data Set
    :param filename: path to csv file
    :return: X (data) and y (labels)
    """

    data = pd.read_csv(filename)
    Gender = {'Male': 1, 'Female': 2,'Other': 3}
    evermarried = {'Yes': 1, 'No': 2}
    worktype = {'Private': 1, 'Self-employed': 2}


    data.gender = [Gender[item] for item in data.gender]
    data.ever_married = [evermarried[item] for item in data.ever_married]
    data['bmi'].fillna(data['bmi'].median(), inplace=True)
    data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 0})
    data['smoking_status'] = data['smoking_status'].map({'formerly smoked': 1, 'never smoked': 2,'smokes': 3,'Unknown': 4})
    data['work_type'] = data['work_type'].map({'Private': 1, 'Self-employed': 2,'Govt_job': 3,'children': 4, 'Never_worked':5})

    y = data.stroke
    y = y.values
    y = y.astype(int)
    to_drop = ['id', 'stroke']
    X = data.drop(to_drop, axis=1)
    return X, y

def load_women_diabetes_data(filename):
    """
    Loads the women diabete Data Set
    :param filename: path to csv file
    :return: X (data) and y (labels)
    """

    data = pd.read_csv(filename)
    y = data.Outcome
    y = y.values
    y = y.astype(int)
    to_drop = ['Outcome']
    X = data.drop(to_drop, axis=1)
    return X, y


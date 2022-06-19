import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def dataCleaning(data):
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
    labelsToBeEncoded = ['Sex', 'Embarked']
    for label in labelsToBeEncoded:
        le = LabelEncoder()
        data[label] = le.fit_transform(data[label])

    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data = data.dropna(axis = 0)
    return data
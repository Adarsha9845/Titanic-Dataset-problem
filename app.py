from cv2 import Algorithm
import numpy as np
import pandas as pd
import readData
import dataCleaning
import AIalgorithm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score


path = 'tested.csv'
data = readData.readData(path)
data = dataCleaning.dataCleaning(data)

X = data.drop(['Survived'], axis = 1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = AIalgorithm.AIalgorithm(X_train, X_test, y_train, y_test)

yPredict = model.predict(X_test)

y_predict = np.array([[1.0, 0.0] if i[0] > i[1] else [0.0, 1.0] for i in yPredict])

print(y_predict)
print(y_test)

err = f1_score(y_predict, y_test, average=None)
print(err)








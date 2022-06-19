

import imp
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt

def AIalgorithm(X_train, X_test, y_train, y_test):
    
    model = Sequential()
    model.add(Dense(20, input_dim = 7))
    model.add(Dense(30, activation = 'sigmoid'))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = adam(learning_rate = 1e-3), metrics = ['accuracy', ])  
    history = model.fit(X_train, y_train, epochs = 200)
    plt.plot(history.history['loss'])

    return model

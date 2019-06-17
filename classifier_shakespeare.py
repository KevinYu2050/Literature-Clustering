import reader
import numpy as np
from matplotlib import pyplot as plt 
import tensorflow as tf 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

def lstm_classifier(data_dir, label_dir, to_dir):
    # Create a Keras lstm model
    
    batch_size = 256

    # Read in the local data with Shakespearean content
    x_train, y_train, x_valid, y_valid, vocab = reader.read_data(data_dir, label_dir)

    # Create a linear stack of models
    model = Sequential()
    model.add(Embedding(len(vocab)+1, 300, input_length=15))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=5)


    score = model.evaluate(x_valid, y_valid, batch_size=batch_size)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    
    # Save the model
    model.save(to_dir)
    
    return history

def plot(history, str):
    # A helper function to plot graphs
    plt.plot(history.history[str])
    plt.plot(history.history['val_' + str])
    plt.xlabel('Epochs')
    plt.ylabel(str)
    plt.legend([str, 'val_' + str])
    plt.show()
    
history = lstm_classifier('./data.txt', './label.txt', "./keras_Shakespeare.h5")
plot(history, 'acc')
plot(history, 'loss')




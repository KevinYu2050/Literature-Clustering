import reader
from numpy import loadtxt
from keras.models import load_model
from matplotlib import pyplot as plt 
# import tensorflow as tf 
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
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=5)


    score = model.evaluate(x_valid, y_valid, batch_size=batch_size)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    
    # Save the model
    model.save(to_dir)
    
    return history

def plot(history):
    # A helper function to plot graphs
    for str in history.history.keys():
        fig = plt.figure()
        plt.plot(history.history[str])
        plt.xlabel('Epochs')
        plt.ylabel(str)
        plt.title('model {}'.format(str))
        plt.show()
        fig.savefig('./{}_history'.format(str), dpi=fig.dpi)

def load_model_(data_dir, label_dir, dir):
    # load and evaluate a saved model
    model = load_model(dir)
    # summarize model.
    print(model.summary())
    _, _, x_valid, y_valid, _ = reader.read_data(data_dir, label_dir)
    score = model.evaluate(x_valid, y_valid)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100)) # acc: 94.85%

    
history = lstm_classifier('./data_with_punctuation.txt', './label_with_punctuation.txt', "./keras_Shakespeare.h5")
plot(history)
# load_model_('./data.txt', './label.txt', './keras_Shakespeare.h5')




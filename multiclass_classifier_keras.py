import rnn_reader as reader
from numpy import loadtxt
from matplotlib import pyplot as plt 
import tensorflow as tf 

def lstm_classifier(train_data, test_data, vocab_size, to_dir):
    # Create a Keras lstm model
    

    # Create a linear stack of models
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 256))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, dropout=0.2, recurrent_dropout=0.2)))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.add(tf.keras.layers.Dense(21, activation='softmax')) # output layer
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_data, steps_per_epoch=150000//256, epochs=5, validation_data=test_data,
        validation_steps=50000//256)


    score = model.evaluate(test_data)
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
        fig.savefig('./{}_history_multiclass'.format(str), dpi=fig.dpi)

def load_model_(data_dir, label_dir, dir):
    # load and evaluate a saved model
    model = tf.keras.load_model(dir)
    # summarize model.
    print(model.summary())
    _, _, x_valid, y_valid, _ = reader.read_data(data_dir, label_dir)
    score = model.evaluate(x_valid, y_valid)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100)) # acc: 94.85%


train_data, test_data, vocab_size = reader.read_data('./gutenberg_preprocessed/')
print(train_data, test_data, vocab_size)
history = lstm_classifier(train_data, test_data, vocab_size, "./keras_multiclass.h5")
plot(history)
# load_model_('./data.txt', './label.txt', './keras_Shakespeare.h5')




import rnn_reader as reader
from numpy import loadtxt
from matplotlib import pyplot as plt
import tensorflow as tf 
from keras.models import load_model 

def lstm_classifier(data_dir, to_dir):
    # Create a Keras lstm model
    
    train_data, test_data, vocab = reader.read_data(data_dir)
    train_iter = train_data.make_one_shot_iterator()
    test_iter = test_data.make_one_shot_iterator()
    batch_size = 256
    print('Dataset read.')

    # Create a linear stack of models
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab+1, batch_size)) # Some unknown tf error
    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(batch_size, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(batch_size//2)))
    model.add(tf.keras.layers.Dense(batch_size, activation='relu'))
    model.add(tf.keras.layers.Dense(21, activation='softmax')) # output layer
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
     metrics=['accuracy'])
    print('Model created.')

    history = model.fit(train_iter, epochs=3, validation_data=test_iter,
     steps_per_epoch=500000//batch_size, validation_steps=125000//batch_size)

    score = model.evaluate(test_iter, steps=125000//batch_size)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    
    # Save the model
    model.save(to_dir)
    
    return history, score

def plot(history, score):
    # A helper function to plot graphs
    for str in history.history.keys():
        fig = plt.figure()
        plt.plot(history.history[str])
        plt.xlabel('Epochs')
        plt.ylabel(str)
        plt.title('model {}'.format(str))
        plt.xticks(())
        plt.yticks(())
        plt.text(4.0, 0.91, 'test loss: %.2f' % score[0], size=15,
         horizontalalignment='right')
        plt.text(4.0, 0.90, 'test accuracy: %.2f' % score[1], size=15,
         horizontalalignment='right')
        plt.show()
        fig.savefig('./{}_history_multiclass_with_dense_layers'.format(str),dpi=fig.dpi)

def load_model_(data_dir, label_dir, dir):
    # load and evaluate a saved model
    model = load_model(dir)
    # summarize model.
    print(model.summary())
    _, _, x_valid, y_valid, _ = reader.read_data(data_dir, label_dir)
    score = model.evaluate(x_valid, y_valid)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100)) 

history, score = lstm_classifier('./gutenberg_preprocessed/', 
        "./keras_Multiclass_dense_layers.h5")

plot(history, score)



# load_model_('./data.txt', './label.txt', './keras_Shakespeare.h5')




import rnn_reader_keras as reader
import multiclass_classifier_keras as clf
import tensorflow as tf 

def autoencoder(data_dir, label_dir, to_dir, to_dir_weights):
    # Create an autoencoder

    x_train, _, _, _, vocab = reader.read_data(data_dir, label_dir)
    batch_size = 32

    model = tf.keras.Sequential()
    
    # Embedding layer  
    model.add(tf.keras.layers.Embedding(input_dim=len(vocab)+2, output_dim=256, input_length=100)) # Some unknown tf error
    
    # Encoder 
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu', return_sequences=False)))        
    model.add(tf.keras.layers.RepeatVector(100))        

    # Decoder
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(vocab)+2)))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    print('Model created.')

    history = model.fit(x_train, x_train, epochs=300, batch_size=batch_size)

#     score = model.evaluate(test_iter, steps=125000//batch_size)
#     print('test loss:', score[0])
#     print('test accuracy:', score[1])
    
    # Save the model
    model.save(to_dir)
    model.save_weights(to_dir_weights)
    
    return history, score

history, score = autoencoder(r'C:\Users\kk\Desktop\Pioneer\dataset_for_multiclass_classification.txt',
        r'C:\Users\kk\Desktop\Pioneer\labels_for_multiclass_classification.txt', 
        r'C:\Users\kk\Desktop\Pioneer\keras_autoencoder.h5', 
                r'C:\Users\kk\Desktop\Pioneer\keras_autoencoder_weights.h5')

clf.plot(history, score)

       
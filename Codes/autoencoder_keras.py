import rnn_reader_keras as reader
import multiclass_classifier_keras as clf
import tensorflow as tf 

def autoencoder(data_dir, label_dir, to_dir, to_dir_weights, to_dir_summary):
    # Create an autoencoder

    x_train, _, _, _, vocab, seq_len, _ = reader.read_data(data_dir, label_dir, 200000)
    batch_size = 32
    n_features = 256

    model = tf.keras.Sequential()
    
    # Embedding layer  
    # Of input shape (batch_size, input_length, input_dim)
    model.add(tf.keras.layers.Embedding(input_dim=len(vocab)+1, output_dim=n_features, input_length=seq_len)) # Some unknown tf error
    # Encoder 
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu', return_sequences=False)))        
    model.add(tf.keras.layers.RepeatVector(seq_len))        

    # Decoder 
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True)))
    # Of output shape (batch_size, ,Dense shape)
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(vocab)+1), input_shape=(seq_len, n_features)))
    # model.add(tf.keras.layers.Reshape((15), input_shape=(15, 10004)))

    optimizer = tf.keras.optimizers.Adam(lr=0.1, decay=0.1/30)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    print('Model created.')
    print(model.summary())

    history = model.fit(x_train, x_train, epochs=30, batch_size=batch_size, shuffle=True)

#     print('test loss:', score[0])
#     print('test accuracy:', score[1])
    
    # Save the model
    model.save(to_dir)
    model.save_weights(to_dir_weights)
    # with open(to_dir_summary, 'w+', encoding='utf-8') as f:
    #     f.write(model.summary())
    
    return history

history = autoencoder(r'C:\Users\kk\Desktop\Pioneer\dataset_for_multiclass_classification_test_modified.txt',
        r'C:\Users\kk\Desktop\Pioneer\labels_for_multiclass_classification_test_modified.txt', 
        r'C:\Users\kk\Desktop\Pioneer\keras_autoencoder_modified.h5', 
                r'C:\Users\kk\Desktop\Pioneer\keras_autoencoder_weights_modified.h5',
                  r'C:\Users\kk\Desktop\Pioneer\keras_autoencoder_summary.txt')


clf.plot(history)

       
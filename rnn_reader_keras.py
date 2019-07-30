from keras.preprocessing.sequence import TimeseriesGenerator
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import re
import numpy as np
import string


def read_data(data_dir, label_dir):
    # Generate a Keras sequence object as the data set
    with open(data_dir, encoding='utf8') as f:
        data = f.readlines()
    data = [s.replace('\n', '') for s in data]
    data = [re.sub(' +', ' ', s) for s in data]

    with open(label_dir, encoding='utf8') as f:
        labels = f.readlines()
    labels = [s.replace('\n', '') for s in labels]
    
    # Tokenize the document
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    # Create a dictionary
    vocab = tokenizer.word_index

    # Divide train and test datasets
    x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.33, shuffle= True)

    # Encoding labels
    y_labels = list(set(y_valid))
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y_labels)
    y_train = encoder.transform(y_train)
    y_valid = encoder.transform(y_valid)

    # Padding the sequences
    X_train_word_ids = tokenizer.texts_to_sequences(x_train)
    X_valid_word_ids = tokenizer.texts_to_sequences(x_valid)

    x_train = pad_sequences(X_train_word_ids, maxlen=100)
    x_valid = pad_sequences(X_valid_word_ids, maxlen=100)
    
    return x_train, y_train, x_valid, y_valid, vocab

# read_data(r'C:\Users\kk\Desktop\Pioneer\dataset_for_multiclass_classification.txt',
#         r'C:\Users\kk\Desktop\Pioneer\labels_for_multiclass_classification.txt')
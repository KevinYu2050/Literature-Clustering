import os
import sys   
import tensorflow as tf
import tensorflow_datasets as tfds

sys.setrecursionlimit(250000)

# tf.enable_eager_execution()

def labeler(example, index):
    # A labeler function 
    return example, tf.cast(index, tf.int32)



def read_data(dir):
    # read the data into a Tensorflow Dataset object
    batch_size = 256
    buffer_size = 50000 # Affects the randomness of the dataset
    take_size = 200000
    test_size = 50000

    # A container for all dataset objects
    labeled_datasets = []

    # Create a dataset
    file_names = [f for f in os.listdir(dir) 
        if f.endswith('.txt')]
    
    # Convert labels to integers
    name_dict = list(set([file_name.split(':')[0] for file_name in file_names]))
    
    # Take in all the datasets independently
    for file_name in file_names:
        lines_dataset = tf.data.TextLineDataset(os.path.join(dir, file_name))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, 
        name_dict.index(file_name.split(':')[0])))
        labeled_datasets.append(labeled_dataset)
    # Concatenate all the dataset objects in the container
    dataset = labeled_datasets[0]
    for labeled_dataset in labeled_datasets[1:]:
        dataset = dataset.concatenate(labeled_dataset)
    print('Initial dataset created.')
    
    dataset = dataset.shuffle(
    buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.take(take_size)

    text_tokenizer = tfds.features.text.Tokenizer()
    # label_tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()
    # labels_vocab = set()

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # for text_tensor, _ in dataset:
    #     some_tokens = tokenizer.tokenize(text_tensor.numpy())
    #     vocabulary_set.update(some_tokens)
    # print('Vocabulary created.')

    with tf.Session() as sess:
        while True:
            try:
                text_tensor, _ = sess.run(next_element)
                some_tokens = text_tokenizer.tokenize(text_tensor)
                # some_labels = label_tokenizer.tokenize(text_tensor)
                vocabulary_set.update(some_tokens)
                # labels_vocab.update(some_labels)
            except tf.errors.OutOfRangeError:
                break

    vocab_size = len(vocabulary_set)
    print('Vocabulary created.')


    # Encode the dataset and the labels using dictionaries 
    text_encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    # label_encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    def encode(text_tensor, label_tensor):
        # A encoder function to vectorize the text 
        encoded_text = text_encoder.encode(text_tensor.numpy())
        # encoded_label = label_encoder.encode(label_tensor.numpy())

        return encoded_text, label_tensor

    def encode_map_fn(text, label):
        # A wrapper function 
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int32))

    dataset = dataset.map(encode_map_fn)
    print('Dataset encoded.')

    # Train-test data split
    train_data = dataset.skip(test_size).shuffle(buffer_size)
    train_data = train_data.padded_batch(batch_size, padded_shapes=([None],[]))

    test_data = dataset.take(test_size)
    test_data = test_data.padded_batch(batch_size, padded_shapes=([None],[]))

    vocab_size += 1 # Add the padded shape into the vocabulary


    return train_data, test_data, vocab_size



# train_data, test_data, vocab_size = read_data('./gutenberg_preprocessed/')
# test_iter = test_data.make_one_shot_iterator()
# next_element = test_iter.get_next()
# with tf.Session() as sess:
#     for _ in range(10):
#         text_tensor, label_tensor = sess.run(next_element)
#         print(text_tensor[0], label_tensor[0])
#         print(text_tensor, label_tensor)




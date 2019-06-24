import os
import tensorflow as tf
import tensorflow_datasets as tfds

tf.enable_eager_execution()

def labeler(example, index):
    # A labeler function 
    return example, tf.cast(index, tf.string)



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
    # Take in all the datasets independently
    for file_name in file_names:
        lines_dataset = tf.data.TextLineDataset(os.path.join(dir, file_name))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, file_name.split(':')[0]))
        labeled_datasets.append(labeled_dataset)
    # Concatenate all the dataset objects in the container
    dataset = labeled_datasets[0]
    for labeled_dataset in labeled_datasets[1:]:
        dataset = dataset.concatenate(labeled_dataset)
    
    dataset = dataset.shuffle(
    buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.take(take_size)

    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()

    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    # initializer = iterator.initializer

    for text_tensor, _ in dataset:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    # with tf.Session as sess:
    #     sess.run(initializer)
    #     for _ in range(take_size):
    #         text_tensor, _ = sess.run(next_element)
    #         some_tokens = tokenizer.tokenize(text_tensor.numpy())
    #         vocabulary_set.update(some_tokens)


    vocab_size = len(vocabulary_set)

    # Encode the dataset using the dictionary 
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    example_text = next(iter(dataset))[0].numpy()

    def encode(text_tensor, label):
        # A encoder function to vectorize the text 
        encoded_text = encoder.encode(text_tensor.numpy())

        return encoded_text, label

    def encode_map_fn(text, label):
        # A wrapper function 
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.string))

    dataset = dataset.map(encode_map_fn)
    
    # Train-test data split
    train_data = dataset.skip(test_size).shuffle(buffer_size)
    train_data = train_data.padded_batch(batch_size, padded_shapes=([-1],[]))

    test_data = dataset.take(test_size)
    test_data = test_data.padded_batch(batch_size, padded_shapes=([-1],[]))

    vocab_size += 1 # Add the padded shape into the vocabulary


    return train_data, test_data



# read_data('./gutenberg_preprocessed/')

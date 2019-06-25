import rnn_reader as reader 
import numpy as np 
import tensorflow as tf 

class Nn_config:
    """configurations for the model."""

    def __init__(self, num_classes, vocab_size, hidden_size, init_scale, batch_size, 
                num_steps, num_layers, is_training, keep_prob, learning_rate, max_lr_epoch, 
                lr_decay, display_freq):

        self.num_classes = num_classes # number of classes for classification

        self.vocab_size = vocab_size # size of vocabulary 
        self.hidden_size = hidden_size # dimensionality of embeddings,with the embedding layer 
                                            # having the shape of vocab_size * hidden_size
        self.init_scale = init_scale # range of the uniform distribution for the embedding layer
        
        self.batch_size = batch_size 
        self.num_steps = num_steps # the number of training steps
        self.num_layers = num_layers
        self.is_training = is_training # allows the model instance to be created either as a model setup 
                                    # for training, or alternatively setup for validation or testing only.
        self.keep_prob = keep_prob  # dropout rate
        self.learning_rate = learning_rate
        self.max_lr_epoch = max_lr_epoch
        self.lr_decay = lr_decay # time-based decay for learning rate
        self.display_freq = display_freq # frequency of displaying the results


class Rnn:
    """the RNN model"""

    def __init__(self, config, input_):
        self.num_classes = config.num_classes

        self.vocab_size = config.vocab_size 
        self.hidden_size = config.hidden_size 
        self.init_scale = config.init_scale
        self.keep_prob = config.keep_prob  


        self.batch_size = config.batch_size 
        self.num_steps = config.num_steps
        self.num_layers = config.num_layers
        self.is_training = config.is_training

        self.learning_rate = config.learning_rate
        self.max_lr_epoch = config.max_lr_epoch
        self.lr_decay = config.lr_decay 
        self.display_freq = config.display_freq

        self.input = input_

    def embedding_layer(self):
        # Create the embedding layer

        embedding = tf.get_variable("encoder",
                                    [self.vocab_size, self.hidden_size],
                                    dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.init_scale, self.init_scale))
        inputs = tf.nn.embedding_lookup(embedding, self.input.data)
        # Apply a dropout wrapper to prevent overfitting
        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)
        
        return inputs

    def lstm_cell(self):
        # Create the LSTM network

        # Create the state variable
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])
        # the size of variable: (num_layers, 2, self.batch_size, self.hidden_size)
        # num_layers: we need a state variable for each layer 
        # 2: previous state variable and previous output in the LSTM network

        # Set up the state data variable in the format required by TensorFlow LSTM data structure
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(self.num_layers)]
        )
        # Tf.unstack creates a number of tensors of size (2, self.batch_size, self.hidden_size), 
        # who are then loaded into LSTMStateTuple objects for LSTM input

        # Create an LSTM cell 
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0)
        # Apply a dropout wrapper to prevent overfitting
        if self.is_training and self.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        
        # Use MultiRNNCell object to include multiple cells
        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.num_layers)], state_is_tuple=True)
        
        return cell, rnn_tuple_state

    def dynamic_rnn(self):
        # Create a dynamic Tensorflow RNN object, which will dynamically perform the unrolling of the LSTM cell over each time step
        
        cell, rnn_tuple_state = self.lstm_cell()
        inputs = self.embedding_layer()

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # The dynamic_rnn object takes the defined LSTM cell as the first argument,
        # the embedding tensor inputs as the second,
        # and initial_state, where the time-step zero state variables are loaded, as the third

        return output

    def loss(self):
        # Set up the softmax weight variables 
        output = self.dynamic_rnn()

        softmax_w = tf.Variable(tf.random_uniform([self.hidden_size, self.vocab_size], -self.init_scale, self.init_scale))
        softmax_b = tf.Variable(tf.random_uniform([self.vocab_size], -self.init_scale, self.init_scale))
        # The softmax operation is yet to be added, this is simply the output of tensor multiplication
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b) 

        # Define the loss function for classification
        # Seq2seq loss function(weighted cross entropy loss over a sequence of values)
        # the logits tensor(1st argument) requires tensors with shape (batch_size, num_steps, vocab_size)
        # the targets tensor(2nd argument) with shape (batch_size, num_steps) and each value being an integer 
        # weights tensor(3rd argument) of shape (batch_size, num_steps), which allows you to weight different samples or time steps with respect to the loss

        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
                    logits,
                    self.input.targets,
                    tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
                    average_across_timesteps=False,
                    average_across_batch=True)
        # If average_across_timesteps is set to True, the cost will be summed across the time dimension, 
        # If average_across_batch is True, then the cost will be summed across the batch dimension

        # Update the cost
        self.cost = tf.reduce_sum(loss)

        return logits
    
    def predict(self):
        # Get the prediction accuracy
        logits = self.optimizer()
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, self.vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # First apply a softmax operation to get the predicted probabilities of each word for each output 
        # Then make the network predictions equal to those words with the highest softmax probability
        # These predictions are then compared to the actual target words and then averaged to get the accuracy

    def optimizer(self):
        if not is_training:
            return

        self.learning_rate = tf.Variable(0.0, trainable=False) # create a learning rate variable if the model is training

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)  # clip the size of the gradients in our network during back-propagation
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate) # create the optimizer operation and apply the clipped gradients
        # Gradient descent 
        self.train_op = optimizer.apply_gradients(
                    zip(grads, tvars),
                    global_step=tf.contrib.framework.get_or_create_global_step())
        # Update the learning rate
        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)
    

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

def train(config, train_data):
        # train the data

        # Set up data and models
        model = Rnn(config, train_data)

        # Set up the training data
        iter = train_data.make_one_shot_iterator()
        x, y = iter.get_next()
        init_op = tf.global_variables_initializer()

        orig_decay = config.lr_decay
        with tf.Session() as sess:
            # start threads
            sess.run([init_op])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            saver = tf.train.Saver()
            









train_data, test_data, vocab_size = reader.read_data('./gutenberg_preprocessed/')

nn_config_train = Nn_config(num_classes = 21, vocab_size=vocab_size, hidden_size=700, init_scale=0.05, batch_size=256, 
                num_steps=40, num_layers=3, is_training=True, keep_prob=0.5, learning_rate=1.0, max_lr_epoch=10, 
                lr_decay=0.93, display_freq=20)
















import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

class LSTMModel():
    """
    To construct and train an LSTM tensorflow model
    """

    @classmethod
    def create_from_args(cls, is_sample_mode=False):
        """
        Create an instance of this class from the passed arguments
        :param args: object from argparse
        :return: instance of TrainModel
        """
        args = cls.parse_args()
        return LSTMModel(args.input, args.iterations, args.state_size, args.lr, args.sample_every, args.sample_size, is_sample_mode, args.first_value)

    @classmethod
    def parse_args(cls):
        """
        Parse arguments on the command line
        :param args:
        :return:
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('-iterations', help='Number of iterations to train the network', type=int, default=35000)
        parser.add_argument('-state_size', help='State size of the lstm cell', type=int, default=5)
        parser.add_argument('-lr', help='Learning rate ', type=float, default=0.05)
        parser.add_argument('-sample_every', help='Sample during training', type=int, default=0)
        parser.add_argument('-sample_size', help='Number of words to sample', type=int, default=10)
        parser.add_argument('-input', help='File containing sequence of words', required=True)
        parser.add_argument('-first_value', help='First character of the sample data in sample_mode', type=str)

        return parser.parse_args()

    def __init__(self, input_file, iterations, state_size, learning_rate, sample_every, sample_size, is_sample_mode, first_value):
        """

        :param filename:
        :param iterations:
        :param state_size:
        :param learning_rate:
        :param sample_every:
        :param sample_size:
        """
        self.input_file = input_file
        self.iterations = iterations
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.sample_every = sample_every
        self.sample_size = sample_size
        self.batch_size = 1
        self.is_sample_mode = is_sample_mode

        print("Configuration used")
        print("Input used", self.input_file)
        print("Number of iterations ", self.iterations)
        print("Learning rate ", self.learning_rate)
        print("State size of lstm", self.state_size)
        print("Sample every %d step %d word" % (self.sample_every, self.sample_size))
        print("Batch size:", 1)

        #Length of the sequences will be determined runtime
        self.length = None
        self.X = tf.placeholder(tf.int32, [self.batch_size, self.length])
        self.Y = tf.placeholder(tf.int32, [self.batch_size, self.length])

        data = open(self.input_file, 'r').read().lower()
        chars = list(set(data))
        self.num_classes = self.vocab_size = len(chars)

        self.char_to_class_id = {ch: i for i, ch in enumerate(sorted(chars))}
        self.class_id_to_char = {i: ch for i, ch in enumerate(sorted(chars))}

        if first_value is not None:
            if first_value not in self.char_to_class_id:
                raise ValueError('First value must be in dictionary')
            if not self.is_sample_mode:
                raise ValueError('First value can only be specified in sample mode')
            self.first_value = self.char_to_class_id[first_value]
        else:
            self.first_value = -1

    def create_graph(self, num_classes):
        """
        Constructs the graph of the model
        Creates the placeholder for the hidden state and cell state
        batch_size - Number of data rows
        state_size - state_size of the RNN Unit
        num_classes - number of classes we are predicting
        parameters - matrices containing weights

        return cell_state, hidden_state, current_state, predictions, total_loss
        """
        # Create one hot representaion from input X placeholder
        inputs_series = tf.one_hot(self.X, num_classes)

        # Create lstm_cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True)

        # Create placeholder for cell state and hidden state
        self.cell_state = tf.placeholder(tf.float32, [self.batch_size, self.state_size])
        self.hidden_state = tf.placeholder(tf.float32, [self.batch_size, self.state_size])
        # LSTMStateTuple represent the state of the cell
        rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(self.cell_state, self.hidden_state)

        # Unroll the cell to a max_length connected rnn cells
        # The length of the timeseries is dynamically determined during runtime, as every datarow has different length
        outputs, current_state = tf.nn.dynamic_rnn(lstm_cell,
                                                   inputs_series,
                                                   initial_state=rnn_tuple_state, dtype=tf.float32)
        # Determine length of sequence
        length = tf.shape(self.X)[1]

        # outputs will have a shape of batch_size X state_size
        # we define and out matrix of shape state_size, num_classes
        # outputs * out_weight will result in an output of the desired shape
        out_weight = tf.get_variable('out_weight', [self.state_size, num_classes])
        out_bias = tf.get_variable('out_bias', [num_classes])

        logits = tf.reshape(tf.matmul(tf.reshape(outputs, [-1, self.state_size]), out_weight) + out_bias,
                            [self.batch_size, length, num_classes])

        # Create prediction for sampling purposes
        self.predictions = tf.nn.softmax(logits)
        # Calculate loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y, logits=logits)
        # Calculate total loss as average over timesteps
        total_loss = tf.reduce_mean(losses)

        return current_state, total_loss

    def sample(self, sess, current_state, sample_size):
        for sample in range(sample_size):
            _current_sample_state = np.zeros((2, self.batch_size, self.state_size))
            # sample a prediction
            idx = self.first_value
            newline_character = self.char_to_class_id['\n']
            counter = 0
            indices = []
            if idx != -1:
                indices.append(idx)
            X_eval = [idx]
            X_eval = np.expand_dims(np.array(X_eval), axis=0)
            while (idx != newline_character and counter != 50):
                #                         np.random.seed(counter+sample)
                pred_out, _current_sample_state = sess.run([self.predictions, current_state],
                                                           feed_dict={
                                                               self.X: X_eval,
                                                               self.cell_state: _current_sample_state[0],
                                                               self.hidden_state: _current_sample_state[1]})
                pred_probs = pred_out[0][0]

                # Sample a character using the output probability distribution
                idx = np.random.choice(np.arange(0, self.vocab_size), p=pred_out.ravel())
                # Append sampled character to a list
                character = self.class_id_to_char[idx]
                indices.append(idx)
                # set sampled characted as an input in the next timestep
                X_eval = [idx]
                X_eval = np.expand_dims(np.array(X_eval), axis=0)
                counter += 1
            print(''.join([self.class_id_to_char[i] for i in indices]).strip())

    def train(self, sess, current_state, total_loss, train_step):
        with open(self.input_file) as f:
            datarows = f.readlines()
        datarows = [x.lower().strip() for x in datarows]
        np.random.shuffle(datarows)

        for step in range(1, self.iterations + 1):
            # Zero Initialize the hidden and cell state of the lstm
            _current_state = np.zeros((2, self.batch_size, self.state_size))
            row_index = step % len(datarows)

            X_train = [-1] + [self.char_to_class_id[ch] for ch in datarows[row_index]]
            Y_train = X_train[1:] + [self.char_to_class_id["\n"]]

            # Reshape data to get 1x28 shaped element
            batch_x = np.expand_dims(np.array(X_train), axis=0)
            batch_y = np.expand_dims(np.array(Y_train), axis=0)

            cost, _current_state, _ = sess.run([total_loss, current_state, train_step],
                                               feed_dict={
                                                   self.X: batch_x,
                                                   self.Y: batch_y,
                                                   self.cell_state: _current_state[0],
                                                   self.hidden_state: _current_state[1]})

            # Print Loss and sample from trained grapd
            if self.sample_every != 0 and step % self.sample_every == 0 or self.iterations == step:
                print("Step " + str(step) + ", Loss= " + "{:.4f}".format(cost))
                self.sample(sess, current_state, self.sample_size)

    def run(self):
        if self.batch_size != 1:
            raise ValueError("batch_size greater then 1 not supported yet")

        current_state, total_loss = self.create_graph(self.num_classes)

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            if(self.is_sample_mode):
                saver.restore(sess, "saved_model/model")
                self.sample(sess,current_state,self.sample_size)
            else:
                # Initialize variables
                sess.run(tf.global_variables_initializer())
                self.train(sess, current_state, total_loss, train_step)
                saver.save(sess, "saved_model/model")
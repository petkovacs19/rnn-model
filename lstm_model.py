import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

class LSTMModel():
    """
    To construct and train an LSTM tensorflow model
    """

    @classmethod
    def create_from_args(cls):
        """
        Create an instance of this class from the passed arguments
        :param args: object from argparse
        :return: instance of TrainModel
        """
        args = cls.parse_args()
        return LSTMModel(args.input, args.iterations, args.state_size, args.lr, args.sample_every, args.sample_size)

    @classmethod
    def parse_args(cls):
        """
        Parse passe arguments on the command line
        :param args:
        :return:
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('-iterations', help='Number of iterations to train the network', type=int, default=35000)
        parser.add_argument('-state_size', help='State size of the lstm cell', type=int, default=5)
        parser.add_argument('-lr', help='Learning rate ', type=float, default=0.05)
        parser.add_argument('-sample_every', help='Sample during training', type=int, default=0)
        parser.add_argument('-sample_size', help='Number of words to sample', type=int, default=0)
        parser.add_argument('-input', help='File containing sequence of words', required=True)

        return parser.parse_args()

    def __init__(self, input_file, iterations, state_size, learning_rate, sample_every, sample_size):
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
        print("Configuration used")
        print("Input used", self.input_file)
        print("Number of iterations ", self.iterations)
        print("Learning rate ", self.learning_rate)
        print("State size of lstm", self.state_size)
        print("Sample every %d step %d word" % (self.sample_every, self.sample_size))
        print("Batch size:", 1)

    def run(self):
        print("yess")






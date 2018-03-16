# rnn-model
A Recurrent Neural Network for training and sampling character-level language models in Tensorflow.

In the example below we use a list of dutch cities as input and we generate new city names by learning the character level patterns in the existing names.
The model generates new sequences of characters using the patterns in the input sequence.

## Usage
The model has been implemented using python and tensorflow and structured to two parts: training and sampling.

If you want to dig deep in how the model works a jupyter notebook is provided that gives a step by step description of the implementation, and an overview of the LSTM(Long Short Term Memory) cell.

If you want to experiment with the different hyperparameters and your own dataset I would recommend to use the python script files.


### Training
First you will need to train the network using train.py.

The following command will run the training on the list of cities in nl_cities.csv, for 35000 iterations with a learning rate of 0.05.
-state_size specifies the state size of the LSTM - Long Short Term Memory Cell
```
python train.py input='nl_cities.csv' -iterations 35000 -state_size 5 -lr 0.05
```

During training I would recommend to keep track of the value of the cost, and sample every x iteration. 
Here is an example of sampling 10 words at every 5000 step. This will print the cost too.

```
python train.py -iterations 35000 -state_size 5 -lr 0.05 -sample_every 5000 -sample_size 10
```

When the training has finished, the trained parameters are save to disk to `weigths/` folder.


### Sampling

You can use previously trained weigths to initialize your network and sample words from it.
The following command loads the trained weights and sample 100 words.

```
python sample.py -sample_size 100
```

Here is another example with the first character specified as 'a'

```
python sample.py -sample_size 100 -first_value='a'
```

The sampling works the following way: 
1. The cell state of the lstm is zero initialized. 
2. The first character and the cell state is fed into the lstm cell and generates a probabily distribution for the next character. 
3. We sample a character from the distribution, get the current state of the lstm cell and we feed these to the cell again.
4. Repeat step 3. until we sample a new line character or the max length 50 is reached


## Requirements

All dependencies have been defined in ```requirements.txt```

You can install the dependencies with the following command:

```pip3 install -r requirements.txt```

## References

The implementation has been inspired by Andrew Ng's course of Sequence Models on coursera. I would definitely recommend this course if you want to get a good understanding of sequence models. I would also recommend [colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). He has done an excellent description of LSTMs.



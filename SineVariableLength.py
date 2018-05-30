"""
    Modelling of time series with different lengths example.
    
    Supplementary code for:
    D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"
    """

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate_toy_time_series(sequences = 10, max_length = 100, gap = 5, noise = 0.1, cut_sequence = True):
    """
        Generate toy time series data set, where the time series can have different lengths
        :param sequences: number of time series to generate
        :param max_length: maximum sequence length
        :param gap: prediction horizon (how many steps the labels are in the future)
        :param noise: standard deviation of additive Gaussian noise
        :param cut_sequence: if true, the sequences have variable lengths
        :return: input values, target values, and lenths of time series
        """   
    X = np.empty((sequences, max_length))
    Y = np.empty((sequences, max_length))
    sequence_lengths = np.empty(sequences, dtype=int)
    for i in range(sequences):
        t = np.arange(0, max_length + gap) 
        t0 = np.random.rand() * max_length      
        f = np.random.rand() * 2
        x = np.sin(t + t0)
        X[i,:] = x[0:max_length]
        Y[i,:] = x[gap:max_length+gap] + np.random.normal(0, noise, max_length)
        if cut_sequence:
            length = np.random.randint(max_length/2,max_length) + 1;
            sequence_lengths[i] = length
            X[i,length:] = np.zeros(max_length-length)
            Y[i,length:] = np.zeros(max_length-length)
        else:
            sequence_lengths[i] = max_length
    return X.reshape((sequences, max_length, 1)), Y.reshape((sequences, max_length, 1)), sequence_lengths

def generate_plot(time_series, filename):
    """
        Plots time series
        :param time_series: list of three arrays of time series
        :param filename: name of the image
        """
    fig, ax = plt.subplots(time_series[0].shape[0], sharex=True)
    for i in range(0, time_series[0].shape[0]):
        ax[i].plot(time_series[0][i], label='input', color='lightgray', linestyle='--')
        ax[i].plot(time_series[1][i], label='target')
        ax[i].plot(time_series[2][i], label='model', color='red', linestyle=':')
        ax[i].set_yticks([0])
        ax[i].set_yticks([0])
    plt.xlabel('time [t]')
    plt.ylabel('signal')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


# basic definitions
N = 32  # size of recurrent neural network
T = 100  # maximum length of training time series
n = 10  # number of training sequences
n_test = 2 # number of test sequences
m = 1  # output dimension
d = 1  # input dimension
epochs = 200  # maximum mnumber of training epochs
learning_rate = 0.1  # learning rate

# generate data
train_X, train_Y, train_lengths = generate_toy_time_series(sequences = n, max_length = T, gap = 5, noise = 0.5, cut_sequence=True)
test_X, test_Y, test_lengths = generate_toy_time_series(sequences = n_test, max_length = T, gap = 5, noise = 0.5, cut_sequence=True)

# placeholders
inputs  = tf.placeholder(tf.float32, [None, None, d])
target = tf.placeholder(tf.float32, [None, None, m])
lengths = tf.placeholder(tf.int64)

# network architecture
cell = tf.nn.rnn_cell.GRUCell(N)
rnn_output, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=lengths,
                                  dtype=tf.float32)

# note the following reshaping:
#   We want a prediction for every time step.
#   Weights of fully connected layer should be the same (shared) for every time step.
#   This is achieved by flattening the first two dimensions.
#   Now all time steps look the same as individual inputs in a batch fed into a feed-forward network.
rnn_output_flat = tf.reshape(rnn_output, [-1, N])
target_flat = tf.reshape(target, [-1, m])
prediction_flat = tf.layers.dense(rnn_output_flat, m, activation=None)
prediction  = tf.reshape(prediction_flat, [-1, tf.shape(inputs)[1], m])

# error function
loss = tf.reduce_sum(tf.square(target_flat - prediction_flat)) \
    / tf.cast(tf.reduce_sum(lengths), tf.float32)

# optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# create graph and initialize variables
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # graph is read-only after this statement

    # do the learning
    for i in range(epochs):
        sess.run(train_step,
                 feed_dict={inputs: train_X, target: train_Y,
                            lengths: train_lengths})
        if i==0 or (i+1)%100==0:
            temp_loss = sess.run(loss,
                                 feed_dict={inputs: train_X,
                                            target: train_Y,
                                            lengths: train_lengths})
            print(i+1, 'loss =', temp_loss)

    # visualize modelling of training data
    model = sess.run(prediction, feed_dict={inputs: train_X, lengths: train_lengths})
    generate_plot([train_X, train_Y, model], 'sineTrain.pdf')

    # visualize modelling of test data
    model = sess.run(prediction, feed_dict={inputs: test_X, lengths: test_lengths})
    generate_plot([test_X, test_Y, model], 'sineTest.pdf')


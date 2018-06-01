"""Time series modelling example.

    Supplementary code for:
    D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"
    """
import numpy as np
import tensorflow as tf

# Parameters
gap = 5  # Time steps to predict into the future
T = 500  # Length of training time series
N = 32  # Size of recurrent neural network
n = 1  # Number of training sequences
n_test = 1  # Number of test sequences
m = 1  # Output dimension
d = 1  # Input dimension
epochs = 200  # Maximum number of learning epochs
lr = 0.05  # Learning rate

# Load and arrange data
raw_data = np.genfromtxt('data/lorenz1000.dt')
train_X = raw_data[0:T]
train_Y = raw_data[0+gap:T+gap]
test_X = raw_data[T:-gap]
test_Y = raw_data[T+gap:]
train_X.resize(n, train_X.size, d)
train_Y.resize(n, train_Y.size, m)
test_X.resize(n_test, test_X.size, d)
test_Y.resize(n_test, test_Y.size, m)

# Placeholders
inputs  = tf.placeholder(tf.float32, [None, None, d])
targets = tf.placeholder(tf.float32, [None, None, m])

# Network architecture
cell = tf.nn.rnn_cell.GRUCell(N)  # A recurrent layer of N units
rnn_output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

# Note the following reshaping:
#   We want a prediction for every time step.
#   Weights of fully connected layer should be the same (shared) for every time step.
#   This is achieved by flattening the first two dimensions.
#   Now all time steps look the same as individual inputs in a batch fed into a feed-forward network.
rnn_output_flat = tf.reshape(rnn_output, [-1, N])
targets_flat = tf.reshape(targets, [-1, m])
prediction_flat = tf.layers.dense(rnn_output_flat, m, activation=None)
prediction = tf.reshape(prediction_flat, [-1, tf.shape(inputs)[1], m])

# Error function and optimizer
loss = tf.losses.mean_squared_error(targets_flat, prediction_flat)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Create session and initialize variables
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # Graph is read-only after this statement.

    # Do the learning
    for i in range(epochs):
        sess.run(
            train_step, feed_dict={inputs: train_X, targets: train_Y})
        if (i+1)%10==0:
            print(i+1, ' loss =', sess.run(loss, feed_dict={inputs: train_X, targets: train_Y}))

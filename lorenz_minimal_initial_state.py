"""Time series modelling example showing how to set initial state.
    
    Supplementary code for:
    D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"
    """
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_plot(time_series, filename, plot_input=False):
    """
        Plots time series
        :param time_series: list of three arrays of time series
        :param filename: name of the image
        """
    if plot_input:
        plt.plot(time_series[0][0,:,0], label='input', color='lightgray', linestyle=':', linewidth=3)
    plt.rc('font', size=14)
    plt.plot(time_series[1][0,:,0], label='target', linestyle='-', linewidth=3)
    plt.plot(time_series[2][0,:,0], label='model', linestyle='-', linewidth=3)
    plt.legend(loc=9)
    plt.xlabel('time [t]')
    plt.ylabel('signal')
    plt.savefig(filename)
    plt.close()

# Parameters
gap = 5  # Time steps to predict into the future
T = 500  # Length of training time series
N = 32  # Size of recurrent neural network
n_train = 1  # Number of training sequences
n_test = 1  # Number of test sequences
m = 1  # Output dimension
d = 1  # Input dimension
epochs = 200  # Number of training epochs
lr = 0.05  # Learning rate

# Load and arrange data
raw_data = np.genfromtxt('data/lorenz1000.dt')
train_X = raw_data[0:T]
train_Y = raw_data[0+gap:T+gap]
test_X = raw_data[T:-gap]
test_Y = raw_data[T+gap:]
train_X.resize(n_train, train_X.size, d)
train_Y.resize(n_train, train_Y.size, m)
test_X.resize(n_test, test_X.size, d)
test_Y.resize(n_test, test_Y.size, m)

# Baselines
# Predicting zero
prediction = np.zeros((n_train, train_Y.size, m))
mse = ((train_Y - prediction) ** 2).mean()
print("predicting zero, train: ", mse)
# Predicting mean
prediction = np.zeros((n_train, train_Y.size, m)) + train_Y.mean()
mse = ((train_Y - prediction) ** 2).mean()
print("predicting mean, train: ", mse)
# Predicting previous input
prediction = np.zeros((n_train, train_Y.size, m))
prediction[:,1:,:] = train_Y[:,:-1,:]
mse = ((train_Y - prediction) ** 2).mean()
print("predicting previous input, train: ", mse)

# Placeholders
inputs  = tf.placeholder(tf.float32, [None, None, d])
targets = tf.placeholder(tf.float32, [None, None, m])

# Aetwork architecture
cell = tf.nn.rnn_cell.GRUCell(N)

# A state with all variables set to zero
zero_state = cell.zero_state(tf.shape(inputs)[0], tf.float32)
external_state = tf.placeholder_with_default(zero_state, [None, N])

# RNN definition
rnn_output, new_state = tf.nn.dynamic_rnn(
    cell, inputs, initial_state=external_state, dtype=tf.float32)

# Note the following reshaping:
#   We want a prediction for every time step.
#   Weights of fully connected layer should be the same (shared) for every time step.
#   This is achieved by flattening the first two dimensions.
#   Now all time steps look the same as individual inputs in a batch fed into a feed-forward network.
rnn_output_flat = tf.reshape(rnn_output, [-1, N])
targets_flat = tf.reshape(targets, [-1, m])
prediction_flat = tf.layers.dense(rnn_output_flat, m, activation=None)
prediction  = tf.reshape(prediction_flat, [-1, tf.shape(inputs)[1], m])

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
        sess.run(train_step, feed_dict={inputs: train_X, targets: train_Y})
        if (i+1)%10==0:
            temp_loss = sess.run(loss, feed_dict={inputs: train_X, targets: train_Y})
            print(i+1, ' Loss =', temp_loss)

    # Visualize modelling of training data
    model, final_state = sess.run([prediction, new_state],
                                  feed_dict={inputs: train_X})
    generate_plot([train_X, train_Y, model], 'lorenzTrain.pdf')

    # Visualize modelling of test data
    model = sess.run(prediction, feed_dict={inputs: test_X})
    generate_plot([test_X, test_Y, model], 'lorenzTestZero.pdf')

    # Visualize modelling of test data starting from zero state
    model, loss = sess.run([prediction, loss],
                           feed_dict={inputs: test_X, targets: test_Y,
                                        external_state: final_state})
    print("RNN MSE on test set:", loss)
    generate_plot([test_X, test_Y, model], 'lorenzTestFinal.pdf')

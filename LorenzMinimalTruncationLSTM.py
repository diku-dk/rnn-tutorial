"""Training with long time series.
    
    Supplementary code for:
    D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"
    """

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
nest = tf.contrib.framework.nest

# parameters
gap = 5  # time steps to predict into the future
T = 500  # length of training time series
N = 32  # size of recurrent neural network
n = 1  # number of training sequences
n_test = 1  # number of test sequences
m = 1  # output dimension
d = 1  # input dimension
epochs = 200  # maximum number of epochs
lr = 0.01  # learning rate

# load and arrange data
raw_data = np.genfromtxt('data/lorenz1000.dt')
train_X = raw_data[0:T]
train_Y = raw_data[0+gap:T+gap]
test_X = raw_data[T:-gap]
test_Y = raw_data[T+gap:]
train_X.resize(n, train_X.size, d)
train_Y.resize(n, train_Y.size, m)
test_X.resize(n_test, test_X.size, d)
test_Y.resize(n_test, test_Y.size, m)

# placeholders
inputs = tf.placeholder(tf.float32, [None, None, d])
targets = tf.placeholder(tf.float32, [None, None, m])

# network architecture
cell = tf.nn.rnn_cell.BasicLSTMCell(N)

# a state with all variables set to zero
zero_state = cell.zero_state(n, tf.float32)
# state
state =  nest.map_structure(lambda tensor:
                            tf.Variable(tensor, trainable=False),
                            zero_state)


# RNN
rnn_output, new_state = tf.nn.dynamic_rnn(cell, inputs,
                                      initial_state=state,
                                      dtype=tf.float32)

# state update
update_state = nest.map_structure(tf.assign, state, new_state)
update_state = nest.flatten(update_state)
reset_state  = nest.map_structure(tf.assign, state, zero_state)
reset_state  = nest.flatten(reset_state)

with tf.control_dependencies(update_state):  # update_state is already a list
    rnn_output = tf.identity(rnn_output)

# note the following reshaping:
#   We want a prediction for every time step.
#   Weights of fully connected layer should be the same (shared) for every time step.
#   This is achieved by flattening the first two dimensions.
#   Now all time steps look the same as individual inputs in a batch fed into a feed-forward network.
rnn_output_flat = tf.reshape(rnn_output, [-1, N])
prediction_flat = tf.layers.dense(rnn_output_flat, m,
                                  activation=None)
targets_flat = tf.reshape(targets, [-1, m])
prediction  = tf.reshape(prediction_flat, [-1, tf.shape(inputs)[1], m])

# error function and optimizer
loss = tf.losses.mean_squared_error(targets_flat, prediction_flat)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# create session and initialize variables

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # graph is read-only after this statement

    # do the learning
    for i in range(epochs):
        sess.run(reset_state)  # reset at beginning of each time series
        chunk_size = 50
        for chunk_start in range (0, T, chunk_size):
            sess.run(train_step, feed_dict={
                     inputs:
                     train_X[:, chunk_start: chunk_start + chunk_size],
                     targets:
                     train_Y[:, chunk_start: chunk_start + chunk_size]})
        if (i+1)%10==0:
            sess.run(reset_state)
            temp_loss = sess.run(loss, feed_dict={inputs: train_X, targets: train_Y})
            print(i+1, ' loss =', temp_loss)

    # visualize modelling of training data
    sess.run(reset_state)
    model, final_state = sess.run([prediction, new_state],
                                  feed_dict={inputs: train_X})
    plt.plot(train_X[0,:,0], label='input', color='lightgray', linestyle='--')
    plt.plot(train_Y[0,:,0], label='target')
    plt.plot(model[0,:,0], label='model')
    plt.legend(loc=1)
    plt.xlabel('time [t]')
    plt.ylabel('signal')
    plt.title('data presented in one batch')
    plt.savefig('lorenzTrainChunk.pdf')
    plt.show()

    sess.run(reset_state)
    concatenated = []
    for chunk_start in range (0, T, chunk_size):
        sess.run([prediction, new_state], feed_dict={
                 inputs:
                 train_X[:, chunk_start: chunk_start + chunk_size]})
        concatenated.append(prediction)
    plt.plot(train_X[0,:,0], label='input', color='lightgray', linestyle='--')
    plt.plot(train_Y[0,:,0], label='target')
    plt.plot(model[0,:,0], label='model')
    plt.legend(loc=1)
    plt.xlabel('time [t]')
    plt.ylabel('signal')
    plt.title('data presented in chunks')
    plt.savefig('lorenzTrainChunkEvalChunk.pdf')
    plt.show()




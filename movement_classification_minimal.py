"""Time series classification example.

Supplementary code for:
D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"

Data and problem definition taken from:
D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz. "A public domain dataset for human activity recognition using smartphones". In 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), pages 437–442. i6doc.com, 2013.
"""

import tensorflow as tf
import numpy as np

# load data
train_X = np.load('data/UCIHARtrainX.npy')
test_X  = np.load('data/UCIHARtestX.npy')
train_Y = np.load('data/UCIHARtrainY.npy')
test_Y  = np.load('data/UCIHARtestY.npy')

# parameters
n = train_X.shape[0]  # number of training sequences
n_test = train_Y.shape[0]  # number of test sequences
m = train_Y.shape[1]  # output dimension
d = train_X.shape[2]  # input dimension
T = train_X.shape[1]  # sequence length
epochs = 200
lr = 0.01  # learning rate

# placeholders
inputs = tf.placeholder(tf.float32, [None, None, d])
target = tf.placeholder(tf.float32, [None, m])

# network architecture
N_1 = 32  # number of units in first recurrent layer
N_2 = 32  # number of units in second recurrent layer

rnn_layers = [tf.nn.rnn_cell.GRUCell(N) for N in [N_1, N_2]]
rnn_units = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    
rnn_output, _ = tf.nn.dynamic_rnn(rnn_units, inputs, dtype=tf.float32)

# ignore all but the last timesteps
last = tf.gather(rnn_output, T-1, axis=1)

# fully connected layer
logits = tf.layers.dense(last, m, activation=None)
# output mapped to probabilities by softmax
prediction = tf.nn.softmax(logits)
# error function 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
                      (labels=target, logits=logits))
# 0-1 loss; compute most likely class and compare with target
accuracy = tf.equal(tf.argmax(logits,1), tf.argmax(target,1))
# average 0-1 loss
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
# optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# create session and initialize variables
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()
    # do the learning
    for i in range(epochs):
        sess.run(train_step, feed_dict={inputs: train_X, target: train_Y})
        if (i+1)%10==0:
            tmp_loss, tmp_acc = sess.run([loss,accuracy],feed_dict={inputs: train_X, target: train_Y})
            tmp_acc_test = sess.run(accuracy,feed_dict={inputs: test_X, target: test_Y})
            print(i+1, 'Loss:', tmp_loss, 'Accuracy, train:', tmp_acc, ' Accuracy, test:', tmp_acc_test)

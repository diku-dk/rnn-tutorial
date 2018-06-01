"""Time series classification example.

Supplementary code for:
D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"

Data and problem definition taken from:
D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz. "A public domain dataset for human activity recognition using smartphones". In 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), pages 437–442. i6doc.com, 2013.
"""

import tensorflow as tf
import numpy as np

# Load data
train_X = np.load('data/UCIHARtrainX.npy')
test_X  = np.load('data/UCIHARtestX.npy')
train_Y = np.load('data/UCIHARtrainY.npy')
test_Y  = np.load('data/UCIHARtestY.npy')

# Parameters
n = train_X.shape[0]  # Number of training sequences
n_test = train_Y.shape[0]  # Number of test sequences
m = train_Y.shape[1]  # Output dimension
d = train_X.shape[2]  # Input dimension
T = train_X.shape[1]  # Sequence length
epochs = 200
lr = 0.01  # Learning rate

# Placeholders
inputs = tf.placeholder(tf.float32, [None, None, d])
target = tf.placeholder(tf.float32, [None, m])

# Network architecture
N = 64
rnn_units = tf.nn.rnn_cell.GRUCell(N)
rnn_output, _ = tf.nn.dynamic_rnn(rnn_units, inputs, dtype=tf.float32)

# Ignore all but the last timesteps
last = tf.gather(rnn_output, T - 1, axis=1)

# Fully connected layer
logits = tf.layers.dense(last, m, activation=None)
# Output mapped to probabilities by softmax
prediction = tf.nn.softmax(logits)
# Error function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=target, logits=logits))
# 0-1 loss; compute most likely class and compare with target
accuracy = tf.equal(tf.argmax(logits, 1), tf.argmax(target, 1))
# Average 0-1 loss
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
# Optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Create session and initialize variables
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()
    # Do the learning
    for i in range(epochs):
        sess.run(train_step, feed_dict={inputs: train_X, target: train_Y})
        if (i + 1) % 10 == 0:
            tmp_loss, tmp_acc = sess.run([loss, accuracy], feed_dict={inputs: train_X, target: train_Y})
            tmp_acc_test = sess.run(accuracy, feed_dict={inputs: test_X, target: test_Y})
            print(i + 1, 'Loss:', tmp_loss, 'Accuracy, train:', tmp_acc, ' Accuracy, test:', tmp_acc_test)

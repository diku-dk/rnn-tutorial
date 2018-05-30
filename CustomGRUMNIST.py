"""
    Custom GRU that allows easy gate visualization.
    
    Supplementary code for:
    D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"
    """

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf


class CustomGRU(tf.contrib.rnn.RNNCell):

  def __init__(
      self, size,
      activation=tf.tanh,
      initializer=tf.glorot_uniform_initializer()):
    super(CustomGRU, self).__init__()
    self._size = size
    self._activation = activation
    self._initializer = initializer

  @property
  # The state of a GRU is just its last output, which is of
  # the size of the layer.
  def state_size(self):
    return self._size

  @property
  # Our cell returns the GRU state, as well as the values of
  # the reset and update gates. All three tensors are of the
  # size of the GRU layer.
  def output_size(self):
    return (self._size, self._size, self._size)

  def call(self, input_, state):
    # For efficiency, we use a single layer to compute both the
    # reset and update gates. This means we need to split the
    # output of the layer into two tensors.
    gates = tf.layers.dense(
      tf.concat([state, input_], axis=1),
                2 * self._size, tf.nn.sigmoid,
                bias_initializer=tf.constant_initializer(-1.0))
    reset, update = tf.split(gates, 2, axis=1)
    candidate = tf.layers.dense(
      tf.concat([reset * state, input_], axis=1),
                self._size, self._activation)
    new_state = (1.0 - update) * state + update * candidate
    output = (new_state, reset, update)
    return output, new_state


def visualize_gates(mnist, gates):
  fig, ax = plt.subplots(
      gates[0].shape[0], 3, figsize=(4, gates[0].shape[0]))
  ax[0, 0].set_title('Image')
  ax[0, 1].set_title('Reset gate')
  ax[0, 2].set_title('Update gate')
  for row in range(ax.shape[0]):
    ax[row, 0].imshow(mnist.test.images[row].reshape((28, 28)))
    ax[row, 1].imshow(gates[0][row])
    ax[row, 2].imshow(gates[1][row])
  for axes in ax.flatten():
    axes.set_xticks([])
    axes.set_yticks([])
  fig.tight_layout()
  fig.savefig('custom_gru.pdf')
  print('Saved custom_gru.pdf')


def main():
  images = tf.placeholder(tf.float32, [None, 28, 28])
  labels = tf.placeholder(tf.float32, [None, 10])
  training = tf.placeholder_with_default(False, [])

  cell = CustomGRU(50)
  (output, reset, update), _ = tf.nn.dynamic_rnn(
      cell, images, dtype=tf.float32)
  logits = tf.layers.dense(output[:, -1], 10)
  loss = tf.losses.softmax_cross_entropy(labels, logits)
  error = tf.reduce_mean(tf.cast(tf.not_equal(
      tf.argmax(labels, 1), tf.argmax(logits, 1)), tf.float32))
  
  optimizer = tf.train.AdamOptimizer(3e-4)
  gradients, variables = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 100.0)
  optimize = optimizer.apply_gradients(zip(gradients, variables))

  tf.logging.set_verbosity(tf.logging.ERROR)  # suppress warnings about deprecated stuff
  mnist = input_data.read_data_sets('~/.dataset/tf_mnist/', one_hot=True)
  tf.logging.set_verbosity(tf.logging.INFO)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  print('Start training')
  for epoch in range(10):
    for index in range(60000 // 100 // 10):
      batch = mnist.train.next_batch(100)
      sess.run(optimize, {
          images: batch[0].reshape((-1, 28, 28)),
          labels: batch[1],
          training: True})
    train_error = sess.run(error, {
        images: mnist.train.images[:10000].reshape((-1, 28, 28)),
        labels: mnist.train.labels[:10000]})
    test_error = sess.run(error, {
        images: mnist.test.images.reshape((-1, 28, 28)),
        labels: mnist.test.labels})
    message = 'Epoch {} train error {:.2f}% test error {:.2f}%'
    print(message.format(epoch + 1, train_error * 100, test_error * 100))
  gates = sess.run((reset, update), {
      images: mnist.test.images[:5].reshape((-1, 28, 28))})
  visualize_gates(mnist, gates)


if __name__ == '__main__':
  main()

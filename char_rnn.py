"""Character based language modeling with multi-layer GRUs.

To train the model:

  python3 tf_char_rnn.py --mode training \
      --logdir path/to/logdir --corpus path/to/corpus.txt

To generate text from seed words:

  python3 tf_char_rnn.py --mode sampling \
      --logdir path/to/logdir
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import contextlib
import os

import tensorflow as tf


class RecurrentLanguageModel(object):
  """Stacked RNNs trained to predict the next character of text."""

  def __init__(self, num_layers, num_units):
    """Create the model instance."""
    self._cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.GRUBlockCell(num_units)
        for _ in range(num_layers)])
    self._output_layer = tf.layers.Dense(256, None)

  def optimize(self, chunks, length, learning_rate):
    """Perform gradient descent on the data and return the loss."""
    chunks = tf.one_hot(chunks, 256)
    inputs, targets = chunks[:, :-1], chunks[:, 1:]
    hidden, _ = tf.nn.dynamic_rnn(self._cell, inputs, length, dtype=tf.float32)
    logits = self._output_layer(hidden)
    loss = tf.losses.softmax_cross_entropy(targets, logits)
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.control_dependencies([optimize]):
      return tf.identity(loss)

  def generate(self, seed, length, temperature):
    """Generate a new sequence from a provided starting character."""
    _, state = tf.nn.dynamic_rnn(
        self._cell, tf.one_hot(seed[:, :-1], 256), dtype=tf.float32)
    def sample(values, _):
      token, state = values
      token = tf.one_hot(token, 256)
      hidden, new_state = self._cell(token, state)
      logit = self._output_layer(hidden)
      new_token = tf.distributions.Categorical(logit / temperature).sample()
      return tf.cast(new_token, tf.uint8), new_state
    tokens, _ = tf.scan(sample, tf.range(length), (seed[:, -1], state))
    return tf.transpose(tokens, [1, 0])


def chunk_sequence(sequence, chunk_length):
  """Split a sequence tensor into a batch of zero-padded chunks."""
  num_chunks = (tf.shape(sequence)[0] - 1) // chunk_length + 1
  padding_length = chunk_length * num_chunks - tf.shape(sequence)[0]
  padding = tf.zeros(
      tf.concat([[padding_length], tf.shape(sequence)[1:]], 0),
      sequence.dtype)
  padded = tf.concat([sequence, padding], 0)
  chunks = tf.reshape(padded, [
      num_chunks, chunk_length] + padded.shape[1:].as_list())
  length = tf.concat([
      chunk_length * tf.ones([num_chunks - 1], dtype=tf.int32),
      [chunk_length - padding_length]], 0)
  return tf.data.Dataset.from_tensor_slices((chunks, length))


@contextlib.contextmanager
def initialize_session(logdir):
  """Create a session and saver initialized from a checkpoint if found."""
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  logdir = os.path.expanduser(logdir)
  checkpoint = tf.train.latest_checkpoint(logdir)
  saver = tf.train.Saver()
  with tf.Session(config=config) as sess:
    if checkpoint:
      print('Load checkpoint {}.'.format(checkpoint))
      saver.restore(sess, checkpoint)
    else:
      print('Initialize new model.')
      os.makedirs(logdir, exist_ok=True)
      sess.run(tf.global_variables_initializer())
    yield sess, saver


def training(args):
  """Train the model and frequently print the loss and save checkpoints."""
  dataset = tf.data.TextLineDataset([args.corpus])
  dataset = dataset.map(
      lambda line: tf.decode_raw(line, tf.uint8))
  dataset = dataset.flat_map(
      lambda line: chunk_sequence(line, args.chunk_length))
  dataset = dataset.cache().shuffle(buffer_size=1000).batch(args.batch_size)
  dataset = dataset.repeat().prefetch(1)
  chunks, length = dataset.make_one_shot_iterator().get_next()
  model = RecurrentLanguageModel(args.num_layers, args.num_units)
  optimize = model.optimize(chunks, length, args.learning_rate)
  step = tf.train.get_or_create_global_step()
  increment_step = step.assign_add(1)
  with initialize_session(args.logdir) as (sess, saver):
    while True:
      if sess.run(step) >= args.total_steps:
        print('Training complete.')
        break
      loss_value, step_value = sess.run([optimize, increment_step])
      if step_value % args.log_every == 0:
        print('Step {} loss {}.'.format(step_value, loss_value))
      if step_value % args.checkpoint_every == 0:
        print('Saving checkpoint.')
        saver.save(sess, os.path.join(args.logdir, 'model.ckpt'), step_value)


def sampling(args):
  """Sample text from user provided starting characters."""
  model = RecurrentLanguageModel(args.num_layers, args.num_units)
  seed = tf.placeholder(tf.uint8, [None, None])
  temp = tf.placeholder(tf.float32, [])
  text = tf.concat([seed, model.generate(seed, args.sample_length, temp)], 1)
  with initialize_session(args.logdir) as (sess, saver):
    while True:
      seed_value = [[int(x) for x in input('Seed: ').encode('ascii') or 'We']]
      temp_value = float(input('Temperature: ') or 1.0)
      for text_value in sess.run(text, {seed: seed_value, temp: temp_value}):
        text_value = text_value.tobytes().decode('ascii', 'replace')
        print(text_value)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', choices=['training', 'sampling'])
  parser.add_argument('--logdir', required=True)
  parser.add_argument('--corpus')
  parser.add_argument('--batch_size', type=int, default=100)
  parser.add_argument('--chunk_length', type=int, default=200)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--num_units', type=int, default=500)
  parser.add_argument('--num_layers', type=int, default=3)
  parser.add_argument('--total_steps', type=int, default=100000)
  parser.add_argument('--checkpoint_every', type=int, default=1000)
  parser.add_argument('--log_every', type=int, default=1000)
  parser.add_argument('--sample_length', type=int, default=500)
  args = parser.parse_args()
  if args.mode == 'training':
    training(args)
  if args.mode == 'sampling':
    sampling(args)
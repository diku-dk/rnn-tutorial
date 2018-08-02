# Signal Processing with Recurrent Neural Networks in TensorFlow

## [tensorflow_overview.ipynb](tensorflow_overview.ipynb)

Notebook with the code examples from the TenorFlow introduction Section 2.3.

## [lorenz_minimal.py](lorenz_minimal.py)

Simple example for time series modelling (section 3.2). The RNN is applied to the date stored in [lorenz1000.dt](data/lorenz1000.dt).

## [lorenz_minimal_initial_state.py](lorenz_minimal_initial_state.py)

Simple example for time series modelling (end of section 3.2) showing how to pass initial states to networks with GRU and basic RNN cells. The RNN is applied to the date stored in [lorenz1000.dt](data/lorenz1000.dt). It also generates figures like those in section 3.3.

## [movement_classification_minimal.py](movement_classification_minimal.py)

Simple time series classification example (section 3.3)

The data is stored in [UCIHARTestX.npy](data/UCIHARTestX.npy), [UCIHARTestY.npy](code/UCIHARTestY.npy), [UCIHARTrainX.npy](UCIHARTrainX.npy), and [UCIHARTrainY.npy](UCIHARTrainY.npy)

The example uses a network with several recurrent layers, although this is not necessary. As no regularization is used and the network is highly flexible, overfitting is observed.

The data are taken from:

D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz. A public do- main dataset for human activity recognition using smartphones. In 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), pages 437–442. i6doc.com, 2013.

## [sine_variable_length.py](sine_variable_length.py)

Toy data set showing how to deal with several, variable-length time series (section 4.1).

## [lorenz_minimal_truncation_gru.py](lorenz_minimal_truncation_gru.py)

Example for modelling long time series using GRU and basic RNN cells (section 4.2). The RNN is applied to the date stored in [lorenz1000.dt](data/lorenz1000.dt).

## [lorenz_minimal_truncation_lstm.py](lorenz_minimal_truncation_lstm.py)

Example for modelling long time series using LSTM units (section 4.2). The RNN is applied to the date stored in [lorenz1000.dt](data/lorenz1000.dt).

## [custom_gru_mnist.py](custom_gru_mnist.py)

Example of a custom GRU cell, which also outputs the values of the gates. As a toy example, we consider learning to classify handwritten digits, where the network sees one pixel row per time step.

This example is based on the MNIST data:

Y. LeCun and C. Cortes. [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/).


## [char_rnn.py](char_rnn.py)

Network for character-level language modeling. You have to feed it your own data.

To train the RNN, pass `--mode training` to the script and point to a text file
with one document per line using `--corpus /path/to/corpus.txt`. Moreover, you
need to specify a directory for saving model checkpoints using `--logdir
/path/to/logdir`. Additional options are available to change important 
hyperparameters, such as the batch size, chunk length, learning rate, and network
size.

After training the model, you can generate text from it using `--mode sampling`
and setting `--logdir /path/to/logdir` to point to the saved model.

# Signal Processing with Recurrent Neural Networks in TensorFlow

## [TensorFlowOverview.ipynb](TensorFlowOverview.ipynb)

Notebook with the code examples from the TenorFlow introduction Section 2.3.

## [LorenzMinimal.py](LorenzMinimal.py)

Simple example for time series modelling (section 3.2). The RNN is applied to the date stored in [lorenz1000.dt](data/lorenz1000.dt).

## [LorenzMinimalInitialState.py](LorenzMinimalInitialState.py)

Simple example for time series modelling (end of section 3.2) showing how to pass initial states to networks with GRU and basic RNN cells. The RNN is applied to the date stored in [lorenz1000.dt](data/lorenz1000.dt). It also generates figures like those in section 3.3.

## [MovementClassificationMinimal.py](MovementClassificationMinimal.py)

Simple time series classification example (section 3.3)

The data is stored in [UCIHARTestX.npy](data/UCIHARTestX.npy), [UCIHARTestY.npy](code/UCIHARTestY.npy), [UCIHARTrainX.npy](UCIHARTrainX.npy), and [UCIHARTrainY.npy](UCIHARTrainY.npy)

The example uses a network with several recurrent layers, although this is not necessary. As no regularization is used and the network is highly flexible, overfitting is observed.

The data are taken from:

D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz. A public do- main dataset for human activity recognition using smartphones. In 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), pages 437–442. i6doc.com, 2013.

## [SineVariableLength.py](SineVariableLength.py)

Toy data set showing how to deal with several, variable-length time series (section 4.1).

## [LorenzMinimalTruncationGRU.py](LorenzMinimalTruncationGRU.py)

Example for modelling long time series using GRU and basic RNN cells (section 4.2). The RNN is applied to the date stored in [lorenz1000.dt](data/lorenz1000.dt).

## [LorenzMinimalTruncationLSTM.py](LorenzMinimalTruncationLSTM.py)

Example for modelling long time series using LSTM units (section 4.2). The RNN is applied to the date stored in [lorenz1000.dt](data/lorenz1000.dt).

## [CustomGRUMNIST.py](CustomGRUMNIST.py)

Example of a custom GRU cell, which also outputs the values of the gates. As a toy example, we consider learning to classify  handwritten digits,  where the network sees one pixel row per time step.




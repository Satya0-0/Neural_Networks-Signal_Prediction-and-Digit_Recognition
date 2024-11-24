# Neural_Networks-Signal_Prediction-and-Digit_Recognition

## Data-Driven Insights

This repository contains machine learning models for:

1. Signal Quality Prediction: Predicting signal quality based on various parameters.
2. Digit Recognition: Recognizing digits in images from the SVHN dataset.

## Datasets:

1. Signal Quality: Signals.csv
2. Digit Recognition: svhn.h5

## Model Architecture

Signal Quality Prediction and Digit Recognition:

Neural Network: A feedforward neural network with multiple hidden layers is used.
Activation Function: ReLU, and Sigmoid are used as the activation function in the hidden layers.
Loss Function: Categorical Cross-entropy is used as the loss function.
Optimizer: Adam optimizer is used for optimization.

## Experiment Results

* Signal Quality Prediction:
* 
Performance Metrics: Accuracy
Visualizations: Training and validation loss and accuracy curves

* Digit Recognition:

Performance Metrics: Accuracy
Visualizations: Confusion matrix, sample predictions, Training and validation loss and accuracy curves


## Code Structure

The repository contains only one main part and the code icludes two parts:

notebooks/: Contains the Jupyter Notebook "Signal_Prediction-and-Digit_Recognition.ipynb" for model training and evaluation.

1. PartA_Signal_Quality_Prediction:

data/: Contains the Signals.csv dataset.
model/: Contains the trained model "model" and "model_1".

2. PartB_Digit_Recognition:

data/: Contains the svhn.h5 dataset.
model/: Contains the trained model "model_b".


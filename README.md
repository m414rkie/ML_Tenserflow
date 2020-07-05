# ML_Tensorflow
Forays into machine learning

This repository will hold the codes for the Tensorflow tutorials found at [Tensorflow.org](https://www.tensorflow.org/).

Repository is divided into three main folders. *Basic* holds the introductory tutorials, *Data* holds tutorials on how to handle various data types, and *Estimators* holds tutorials on how to use and customize the Tensorflow premade estimators.

---

## Contents

- [Basic](#basic)
  - [Install Check](#install_check)  
  - [MNIST Numerals](#mnist_numerals)  
  - [MNIST IMDB](#mnist_imdb)  
  - [IMDB Preprocessing](#mnist_imdb_pre)  
  - [Regression](#regression)  
  - [Over/Under Fit](#overunder_fit)
  - [Save and Load](#save_n_load)
  - [HyperTune](#hypertune)

---
## Basic

### Install_check

A short code that verifies installation of Tensorflow and required libraries.

**Contains:**

* first.py
   * Inputs: None
   * Outputs: None
  
 #### MNIST_numerals
 
 The first tutorial. Found [here](https://www.tensorflow.org/tutorials/quickstart/beginner). Tutorial is designed to familiarize the reader with basic functions of Tensorflow. End result is a program that can predict hand-drawn numerals based on grayscale 28X28 pixel images taken from the MNIST dataset. Deviations from the tutorial
 include sections to save various plots, including input data as well as predictions with input data. 
 
 **Contains:**
 * second.py
   * Inputs: None
   * Outputs: 5 plots. One each of accuracy and loss, with three data plots.
   
#### MNIST_imdb

The second tutorial, found [here](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub). This tutorial creates a sentiment predictor for movie reviews based on the IMDB dataset provided with Keras. Text processing from a pre-trained model found [here](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1) Deviations include two new 'reviews' to predict which are admittedly very on the nose. Includes some plots.

**Contains:**
* tf_imdb.py
  * Inputs: None
  * Outputs: 2 plots. One each of accuracy and loss. 

#### MNIST_imdb_pre

The third tutorial, found [here](https://www.tensorflow.org/tutorials/keras/text_classification). This tutorial creates a sentiment predictor for movie reviews based on the IMDB dataset provided with Keras. The text processing uses the TensorFlow Datasets text encoder. Changes from the tutorial include some code that outputs the accuracy and loss in graph form.

**Contains:**
* imdb_2.py
  * Inputs: None
  * Outputs: 2 plots. One each of accuracy and loss

#### Regression

The fourth tutorial, found [here](https://www.tensorflow.org/tutorials/keras/regression). This tutorial predicts the miles-per-gallon of cars from the seventies and eighties. The dataset is the Auto MPG data set and can be found [here](https://archive.ics.uci.edu/ml/datasets/auto+mpg). Tutorial includes some nice visualizations of the data and model error.

**Contains:**
* regression.py
  * Inputs: None
  * Outputs: 5 plots. Data visualization, error bins, MSE and stopped MSE, and prediction vs. true MPG
  
#### OverUnder_fit

The fifth tutorial, found [here](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit). This tutorial deals with the topics of overfitting and uderfitting as well as strategies to reduce these issues. The dataset is the Higgs dataset which contains Monte-Carlo simulations with 21 feature columns and seven label columns. The dataset can be found [here](http://mlphysics.ics.uci.edu/data/higgs/).

**Contains:**
* fitting.py
  * Inputs: None
  * Outputs: 6 plots. 1 of data visualization, 5 of the entropy over epochs, and one of the learning rate.
  
#### Save_N_Load

The fifth tutorial, found [here](https://www.tensorflow.org/tutorials/keras/save_and_load). This tutorial informs the user how to save and load the trained model including checkpoints created during training. Uses the MNIST numerals dataset.

**Contains:**
* save_load.py
  * Inputs: None
  * Outputs: 4 subfolders containing checkpoints, saved models, and 2 training saves. 
  
#### HyperTune

The sixth tutorial, found [here](https://www.tensorflow.org/tutorials/keras/keras_tuner). This tutorial is about how the tune the hyperparameters using inbuilt tuners for both number of neurons and learning rate. Uses the MNIST fashion dataset.

**Contains:**
* HYPER.py
  * Inputs: None
  * Outputs: None

---

  

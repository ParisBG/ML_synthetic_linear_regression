#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 20:20:20 2023
@author: parisbg
"""

import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


'''HYPERPARMATER ADJUSTEMENT RULES OF THUMB'''
#Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
#If the training loss does not converge, train for more epochs.
#If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high may also prevent training loss from converging.
#If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
#Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
#Setting the batch size to a very small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
#For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.
#Remember: the ideal combination of hyperparameters is data dependent, so you must always experiment and verify.


#Define the functions that build and train a model
def build_model(my_learning_rate):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential. 
  # A sequential model contains one or more layers.
  model = tf.keras.models.Sequential()

  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer. 
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

  # Compile the model topography into code that 
  # TensorFlow can efficiently execute. Configure 
  # training to minimize the model's mean squared error. 
  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
  print("**********Defined build_model function.**********", '\n')

  return model           



def train_model(model, feature, label, epochs, batch_size):
  """Train the model by feeding it data."""
  # Feed the feature values and the label values to the model.
  # Model will train for the specified # of epochs
  # Model will gradually learn how the feature values relate to the label values. 
  history = model.fit(x=feature, y=label, batch_size=batch_size, epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]  
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch

  # Gather the history (a snapshot) of each epoch.
  hist = pd.DataFrame(history.history)
  print("\n", 'hist-epochs= ', hist, "\n")

  # Specifically gather the model's root mean 
  # squared error at each epoch. 
  rmse = hist["root_mean_squared_error"]
  print('rmse= ', rmse, "\n")
  
  print("**********Defined train_model function.**********" , '\n')
  return trained_weight, trained_bias, epochs, rmse



#Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
  """Plot the trained model against the training feature and label."""
  # Label the axes.
  plt.xlabel("feature")
  plt.ylabel("label")

  # Plot the feature values vs. label values.
  plt.scatter(feature, label)

  # Create a red line representing the model. The red line starts
  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  x0 = 0
  y0 = trained_bias
  x1 = feature[-1]
  y1 = trained_bias + (trained_weight * x1)  #Remember the ML "slope" formula | y = b + w1 * x1
  plt.plot([x0, x1], [y0, y1], c='r')
  
  # Render the scatter plot and the red line.
  plt.show()
  print("**********Defined the plot_the_model function.**********" , '\n')



def plot_the_loss_curve(epochs, rmse):
  """Plot the loss curve, which shows loss vs. epoch."""
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()
  print("**********Defined plot_the_loss_curve function.**********" , '\n')



''' This dataset consists of 12 examples. Each example consists of one feature and one label. '''
my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])


''' The hyperparameters here are as follows: '''
learning_rate = 0.05 #Ideal = 0.14
epochs = 125 #Ideal = 0.14
my_batch_size = 1

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, 
                                                         my_label, epochs,
                                                         my_batch_size)

plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
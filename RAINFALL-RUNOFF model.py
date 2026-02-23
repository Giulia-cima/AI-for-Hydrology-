# Building a model LSTM for rainfall-runoff prediction using Keras and TensorFlow.
# The model initialli will work with time series data of rainfall as input and runoff/discharge as output

import numpy as np
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Function to create the LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))  # Output layer for runoff prediction
    model.compile(optimizer='adam', loss='mse')
    return model



# upload the input data
precipitation  = pandas.read_pickle("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/mean_h_precipitation_20040901_20230831_cumulative_by_basin.pkl")
runoff = pandas.read_pickle("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/list_q.pkl")

# for each precipitation time series associate the corresponding runoff time series using as reference the name.

# i want to analyze the data , verify if there are clusters of similar time series for entire station, and then create a model for each cluster.



# Preprocessing the data
# Assuming precipitation and runoff are pandas DataFrames with time series data
# remove any missing values
# standardize the data


# # Create the LSTM model



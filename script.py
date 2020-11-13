# frequently used libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


# load test data
test_df = pd.read_csv('test_set.csv')

# fill test data with median 
test_df.fillna(test_df.median(), inplace=True)

# scale the data for enhanced accuracy 
scaler = StandardScaler()
test_trans_df = scaler.fit_transform(test_df)

# load model from .pb file and predict clicks
best_model = tf.keras.models.load_model(r'C:\Users\arali\Desktop\hotel_website')
n_clicks_array = best_model.predict(test_trans_df)

# clicks can not be negative values
n_clicks_array[n_clicks_array < 0] = 0

# number of clicks are integer
floor_n_clicks_array = np.floor(n_clicks_array) 

# create dataframe and save to .csv file
n_clicks_df = pd.DataFrame(floor_n_clicks_array, columns=['predictions'])
n_clicks_df.to_csv('predictions.csv') 
# single var prediction with CNN


import numpy as np
import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sn

from keras.layers import  Conv1D ,Flatten , Dense, MaxPooling1D ,Dropout
from keras.models import Sequential
import keras
import tensorflow as tf


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


msft = pd.read_csv("msft.csv")

msft['Date'] = msft['Date'].astype(str).str[:10]
msft.index = pd.to_datetime(msft['Date'], format='%Y-%m-%d', )

# train = msft[msft.index < pd.to_datetime("2023-06-20", format='%Y-%m-%d')]
# test = msft[msft.index >= pd.to_datetime("2023-06-18", format='%Y-%m-%d')]
#
# X = msft.iloc[:, 5:].values
# y = msft.iloc[:, 4].values

time_series_data = msft['Close'].values.reshape(msft.shape[0])

time_steps = 80
future_steps = 10


# Prepare the input-output pairs using a sliding window approach
X_train = []
y_train = []
for i in range(len(time_series_data) - time_steps):
    X_train.append(time_series_data[i:i+time_steps])
    y_train.append(time_series_data[i+time_steps])


X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(time_steps, 1)))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(future_steps))   # Output layer with one neuron for prediction

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='mse', optimizer=opt )

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=1, verbose=1)


#choosing 0-80 as data, validate the trend of next 20 days

validate_series = time_series_data[0:80]
validate_series = validate_series.reshape(1, time_steps, 1)
predicted_values = model.predict(validate_series)
validate_series = validate_series.reshape(time_steps,1)


predicted_values = predicted_values.reshape(10,1)
validate_series = validate_series.reshape(time_steps,1)
real_values = time_series_data[80:90]
real_values = real_values.reshape(10,1)

predicted_values = predicted_values.reshape(predicted_values.shape[0],1)

x_predict_values = np.arange(80, 80+len(predicted_values))
x_real_values = np.arange(80, 80+len(real_values))

plt.plot(validate_series, color='black', label = 'Predictions')
plt.plot(x_predict_values,predicted_values, color='green', label = 'Predictions')
plt.plot(x_real_values, real_values, color='red', label = 'Predictions')
plt.show()




from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import TimeDistributed  
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("SPY.csv")
data = pd.DataFrame(data)


st = MinMaxScaler(feature_range=(0,1))


# define input sequence (based on open prices)
processed_data = data.iloc[:,1:2].values

#normalize data
processed_data = st.fit_transform(processed_data)


# reshape input into [samples, timesteps, features]

n_in = len(processed_data)
processed_data = np.array(processed_data)
processed_data = processed_data.reshape(1, n_in, 1)

# prepare output sequence

seq_out = processed_data[:, :, :]
n_out = n_in
# define model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_out))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(processed_data, seq_out, epochs=300, batch_size = 256, verbose=1)


#demonstrate prediction
processed_data = model.predict(processed_data, verbose=0)
processed_data = processed_data[0,:,:]
processed_data = st.inverse_transform(processed_data)
print (processed_data)

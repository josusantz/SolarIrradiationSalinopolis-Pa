#SimpleRNN.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, mean_squared_error, explained_variance_score

def calcular_metricas(y_test, y_pred):
    return {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'EVS': explained_variance_score(y_test, y_pred),
    }

df = pd.read_parquet('FormatDataSalinas.gzip')
df = df[(df.index.hour > 11) & (df.index.hour <21)]
df['SMA_5'] = df['GLOBAL RADIATION (Kj/m²)'].rolling(window=5,center =True).mean()
df['windSpeed_km_h'] = df['WIND, HOURLY SPEED (m/s)'] * 3.6
df['ST'] = 33 + (10 * np.sqrt(df['windSpeed_km_h']) + 10.45 - df['windSpeed_km_h']) * (df['AIR TEMPERATURE - DRY BULB, HOURLY (°C)'] - 33) / 22
df.dropna(inplace =True)
y = df['GLOBAL RADIATION (Kj/m²)']
X = df.drop(columns = 'GLOBAL RADIATION (Kj/m²)')

size = .8
train_size = int(len(y)*size)
test_size = int(len(y)-train_size)
print("train size:", train_size)
print("test size:", test_size)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled_rnn = scaler_x.fit_transform(X_train)
X_test_scaled_rnn = scaler_x.transform(X_test)
y_train_scaled_rnn = scaler_y.fit_transform(y_train.to_numpy().reshape(-1,1)).ravel()
y_test_scaled_rnn = scaler_y.transform(y_test.to_numpy().reshape(-1,1)).ravel()


def criarSequencias(X, y, tamanho_sequencia:int):
    antecessores = []
    numero_predito = []
    for i in range(0,len(X)-tamanho_sequencia):
        antecessores.append(X[i:i+tamanho_sequencia])
        numero_predito.append(y[i+tamanho_sequencia])
    return np.array(antecessores), np.array(numero_predito)




X_sequencias_treino, y_sequencias_treino = criarSequencias(X_train_scaled_rnn, y_train_scaled_rnn, 50)
X_sequencias_teste, y_sequencias_teste = criarSequencias(X_test_scaled_rnn, y_test_scaled_rnn, 50)
X_train = np.reshape(X_sequencias_treino, (X_sequencias_treino.shape[0],X_sequencias_treino.shape[1],X_sequencias_treino.shape[2]))
y_train = np.reshape(y_sequencias_treino, (y_sequencias_treino.shape[0],1))
X_test = np.reshape(X_sequencias_teste, (X_sequencias_teste.shape[0],X_sequencias_teste.shape[1],X_sequencias_teste.shape[2]))
y_test = np.reshape(y_sequencias_teste, (y_sequencias_teste.shape[0],1))
print('X_train shape:',X_train.shape, '\ny_train shape:',y_train.shape,'\nX_test shape:',X_test.shape, '\ny_test shape:',y_test.shape)


from keras.optimizers import SGD
# initializing the RNN
regressor = Sequential()

# adding RNN layers and dropout regularization
regressor.add(layers.SimpleRNN(units = 50, 
                        activation = "tanh",
                        return_sequences = True,
                        input_shape = (X_train.shape[1],X_train.shape[2])))
regressor.add(layers.Dropout(0.2))

regressor.add(layers.SimpleRNN(units = 50, 
                        activation = "tanh",
                        return_sequences = True))

regressor.add(layers.SimpleRNN(units = 50,
                        activation = "tanh",
                        return_sequences = True))

regressor.add( layers.SimpleRNN(units = 50))

# adding the output layer
regressor.add(layers.Dense(units = 1,activation='sigmoid'))

# compiling RNN
regressor.compile(optimizer = SGD(learning_rate=0.01,
                                  decay=1e-6, 
                                  momentum=0.9, 
                                  nesterov=True), 
                  loss = "mean_squared_error")

# fitting the model
regressor.fit(X_train, y_train, epochs = 20, batch_size = 2)
regressor.summary()

y_RNN = regressor.predict(X_test)

print('Starting predictions..')
print(y_RNN)

import csv

# Save predictions to a CSV file
csv_file = "/Predicts/Simple_RNN.csv"

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Prediction"])  # Header
    for pred in y_RNN:
        writer.writerow([pred])

print(f"Predictions saved to {csv_file}")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("NASA data for ANN & SVM.csv")
df[:5]

df.info()

df.describe()

plt.figure(figsize=(18,7))
sns.violinplot(x="air temperature", data=df, color = "red")
plt.title("Distribution of Air Temperature", size = 15)
plt.xlabel("Air Temperature", size = 15)
plt.xticks(size = 15)
plt.show()

plt.figure(figsize=(18,7))
sns.violinplot(x="surface humidity", data=df, color = "orange")
plt.title("Distrinution of Surface Humidity", size = 15)
plt.xlabel("Surface Humidity", size = 15)
plt.xticks(size = 15)
plt.show()

plt.figure(figsize=(18,7))
sns.violinplot(x="radiance intensity", data=df, color = "pink")
plt.title("Distrinution of Radiance Intensity", size = 15)
plt.xlabel("Radiance Intensity", size = 15)
plt.xticks(size = 15)
plt.show()

plt.figure(figsize=(18,7))
sns.violinplot(x="surface incomming short wave flux", data=df, color = "lime")
plt.title("Distrinution of Surface Incomming Short Wave Flux", size = 15)
plt.xlabel("Surface Incomming Short Wave Flux", size = 15)
plt.xticks(size = 15)
plt.show()

plt.figure(figsize=(18,7))
sns.violinplot(x="total column ozone", data=df, color = "m")
plt.title("Distrinution of Ozone", size = 15)
plt.xlabel("Ozone", size = 15)
plt.xticks(size = 15)
plt.show()

plt.figure(figsize=(18,7))
sns.violinplot(x="total precipitateable water vapours", data=df, color = "yellow")
plt.title("Distrinution of Total Precipitateable Water Vapours", size = 15)
plt.xlabel("total precipitateable water vapours", size = 15)
plt.xticks(size = 15)
plt.show()

plt.figure(figsize=(18,7))
sns.violinplot(x="wind speed", data=df, color = "skyblue")
plt.title("Distrinution of Wind Speed", size = 15)
plt.xlabel("Wind Speed", size = 15)
plt.xticks(size = 15)
plt.show()

#Festures
X = df[["surface humidity", "radiance intensity", "surface incomming short wave flux", "total column ozone", "total precipitateable water vapours", "wind speed"]]

#Target
y = df["air temperature"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Instantiate Linear SVM Object
svm = SVR()

#Instantiate the GridSearchCV object and run the search
parameter = {'kernel':['linear', 'poly']}
searcher = GridSearchCV(svm, parameter)
searcher.fit(X_train_std, y_train)

#Report the best parameter and the corresponding score
print("Best CV params", searcher.best_params_,"\n")
print("Best CV accuracy", searcher.best_score_)

pred_s1 = searcher.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_s1})[:5]

rmse = MSE(y_test, pred_s1)**(0.5)
print("Root Mean Squared Error =", rmse)

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_s1, label = "Predicted", linewidth = 3, color = "red")
plt.title("Air Temperature Actual and Predicted Values (SVM)", size = 15)
plt.xlabel("Air Temperature", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["surface humidity", "radiance intensity", "surface incomming short wave flux", "total column ozone", "total precipitateable water vapours", "wind speed"]]

#Target
y = df["air temperature"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Initialising the ANN
model = Sequential()

#Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 6))

#Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train_std, y_train, batch_size = 10, epochs = 100)

pred_a1 = model.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_a1.flatten()})[:5]

rmse = MSE(y_test, pred_a1)**(0.5)
print("Root Mean Squared Error =", rmse.round(2))

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_a1.flatten(), label = "Predicted", linewidth = 3, color = "blue")
sns.kdeplot(x = pred_s1, label = "SVM", linewidth = 3, color = "red")
plt.title("Air Temperature Prediction Comparison (SVM & ANN)", size = 15)
plt.xlabel("Air Temperature", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["air temperature", "radiance intensity", "surface incomming short wave flux", "total column ozone", "total precipitateable water vapours", "wind speed"]]

#Target
y = df["surface humidity"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Instantiate Linear SVM Object
svm = SVR()

#Instantiate the GridSearchCV object and run the search
parameter = {'kernel':['linear', 'poly']}
searcher = GridSearchCV(svm, parameter)
searcher.fit(X_train_std, y_train)

#Report the best parameter and the corresponding score
print("Best CV params", searcher.best_params_,"\n")
print("Best CV accuracy", searcher.best_score_)

pred_s2 = searcher.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_s2})[:5]

rmse = MSE(y_test, pred_s2)**(0.5)
print("Root Mean Squared Error =", rmse)

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_s2, label = "Predicted", linewidth = 3, color = "red")
plt.title("Surface Humidity Actual and Predicted Values (SVM)", size = 15)
plt.xlabel("Surface Humidity", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["air temperature", "radiance intensity", "surface incomming short wave flux", "total column ozone", "total precipitateable water vapours", "wind speed"]]

#Target
y = df["surface humidity"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Initialising the ANN
model = Sequential()

#Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 6))

#Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train_std, y_train, batch_size = 10, epochs = 100)

pred_a2 = model.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_a2.flatten()})[:5]

rmse = MSE(y_test, pred_a2)**(0.5)
print("Root Mean Squared Error =", rmse.round(5))

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_a2.flatten(), label = "Predicted", linewidth = 3, color = "blue")
sns.kdeplot(x = pred_s2, label = "SVM", linewidth = 3, color = "red")
plt.title("Precipitateable Water Vapours Prediction Comparison (SVM & ANN)", size = 15)
plt.xlabel("Precipitateable Water", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["surface humidity", "air temperature", "surface incomming short wave flux", "total column ozone", "total precipitateable water vapours", "wind speed"]]

#Target
y = df["radiance intensity"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Instantiate Linear SVM Object
svm = SVR()

#Instantiate the GridSearchCV object and run the search
parameter = {'kernel':['linear', 'poly']}
searcher = GridSearchCV(svm, parameter)
searcher.fit(X_train_std, y_train)

#Report the best parameter and the corresponding score
print("Best CV params", searcher.best_params_,"\n")
print("Best CV accuracy", searcher.best_score_)

pred_s3 = searcher.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_s3})[:5]

rmse = MSE(y_test, pred_s3)**(0.5)
print("Root Mean Squared Error =", rmse)

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_s3, label = "Predicted", linewidth = 3, color = "red")
plt.title("Radiance Intensity Actual and Predicted Values (SVM)", size = 15)
plt.xlabel("Radiance Intensity", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["air temperature", "surface humidity", "surface incomming short wave flux", "total column ozone", "total precipitateable water vapours", "wind speed"]]

#Target
y = df["radiance intensity"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Initialising the ANN
model = Sequential()

#Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 6))

#Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train_std, y_train, batch_size = 10, epochs = 100)

pred_a3 = model.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_a3.flatten()})[:5]

rmse = MSE(y_test, pred_a3)**(0.5)
print("Root Mean Squared Error =", rmse.round(2))

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_a3.flatten(), label = "ANN", linewidth = 3, color = "blue")
sns.kdeplot(x = pred_s3, label = "SVM", linewidth = 3, color = "red")
plt.title("Radiance Intensity Prediction Comparison (SVM & ANN)", size = 15)
plt.xlabel("Radiance Intensity", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["surface humidity", "air temperature", "radiance intensity", "total column ozone", "total precipitateable water vapours", "wind speed"]]

#Target
y = df["surface incomming short wave flux"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Instantiate Linear SVM Object
svm = SVR()

#Instantiate the GridSearchCV object and run the search
parameter = {'kernel':['linear', 'poly']}
searcher = GridSearchCV(svm, parameter)
searcher.fit(X_train_std, y_train)

#Report the best parameter and the corresponding score
print("Best CV params", searcher.best_params_,"\n")
print("Best CV accuracy", searcher.best_score_)

pred_s4 = searcher.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_s4})[:5]

rmse = MSE(y_test, pred_s4)**(0.5)
print("Root Mean Squared Error =", rmse)

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_s4, label = "Predicted", linewidth = 3, color = "red")
plt.title("Surface Incomming Short Wave Flux Actual and Predicted Values (SVM)", size = 15)
plt.xlabel("Surface Incomming Short Wave Flux", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["air temperature", "surface humidity", "radiance intensity", "total column ozone", "total precipitateable water vapours", "wind speed"]]

#Target
y = df["surface incomming short wave flux"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Initialising the ANN
model = Sequential()

#Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 6))

#Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train_std, y_train, batch_size = 10, epochs = 100)

pred_a4 = model.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_a4.flatten()})[:5]

rmse = MSE(y_test, pred_a4)**(0.5)
print("Root Mean Squared Error =", rmse.round(2))

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_a4.flatten(), label = "ANN", linewidth = 3, color = "blue")
sns.kdeplot(x = pred_s4, label = "SVM", linewidth = 3, color = "red")
plt.title("Surface Incomming Short Wave Flux Prediction Comparison (SVM & ANN)", size = 15)
plt.xlabel("Surface Incomming Short Wave Flux", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["air temperature", "surface humidity", "radiance intensity", "surface incomming short wave flux", "total precipitateable water vapours", "wind speed"]]

#Target
y = df["total column ozone"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Instantiate Linear SVM Object
svm = SVR()

#Instantiate the GridSearchCV object and run the search
parameter = {'kernel':['linear', 'poly']}
searcher = GridSearchCV(svm, parameter)
searcher.fit(X_train_std, y_train)

#Report the best parameter and the corresponding score
print("Best CV params", searcher.best_params_,"\n")
print("Best CV accuracy", searcher.best_score_)

pred_s5 = searcher.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_s5})[:5]

rmse = MSE(y_test, pred_s5)**(0.5)
print("Root Mean Squared Error =", rmse)

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_s5, label = "Predicted", linewidth = 3, color = "red")
plt.title("Ozone Actual and Predicted Values (SVM)", size = 15)
plt.xlabel("Ozone", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["air temperature", "surface humidity", "radiance intensity", "surface incomming short wave flux", "total column ozone", "wind speed"]]

#Target
y = df["total precipitateable water vapours"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Initialising the ANN
model = Sequential()

#Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 6))

#Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

#Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train_std, y_train, batch_size = 10, epochs = 100)

pred_b6 = model.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_b6.flatten()})[:5]

rmse = MSE(y_test, pred_b6)**(0.5)
print("Root Mean Squared Error =", rmse.round(3))

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_b6.flatten(), label = "Predicted", linewidth = 3, color = "blue")
sns.kdeplot(x = pred_s6, label = "SVM", linewidth = 3, color = "red")
plt.title("Precipitateable Water Vapours Prediction Comparison (SVM & ANN)", size = 15)
plt.xlabel("Precipitateable Water", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()

#Festures
X = df[["air temperature", "surface humidity", "radiance intensity", "surface incomming short wave flux", "total column ozone", "total precipitateable water vapours"]]

#Target
y = df["wind speed"]

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

#Instantiate Linear SVM Object
svm = SVR()

#Instantiate the GridSearchCV object and run the search
parameter = {'kernel':['linear', 'poly']}
searcher = GridSearchCV(svm, parameter)
searcher.fit(X_train_std, y_train)

#Report the best parameter and the corresponding score
print("Best CV params", searcher.best_params_,"\n")
print("Best CV accuracy", searcher.best_score_)

pred_s7 = searcher.predict(X_test_std)
pd.DataFrame({"Actual": y_test, "Predicted": pred_s7})[:5]

rmse = MSE(y_test, pred_s7)**(0.5)
print("Root Mean Squared Error =", rmse)

plt.figure(figsize=(18,7))
sns.kdeplot(x = y_test, label = "Actual", linewidth = 3, color = "orange")
sns.kdeplot(x = pred_s7, label = "Predicted", linewidth = 3, color = "red")
plt.title("Wind Speed Actual and Predicted Values (SVM)", size = 15)
plt.xlabel("Wind Speed", size = 15)
plt.ylabel("Density", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend()
plt.show()
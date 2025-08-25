import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from funcaoErros import calcular_metricas
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rna_lm import RNALM
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

path = 'dados/dadosLimpos.parquet'

def preprocessar_dados(dados):
    if isinstance(dados, pd.DataFrame):
        None
    else:
         path = dados
         dados = pd.read_parquet(path)
    
   
    #dados.set_index('Data_Hora', inplace = True)
    # df.sort_index(inplace=True)
    dados.dropna(inplace =True)
    dados = dados[(dados.index.hour >= 6) & (dados.index.hour <=18)]
    colunas_araujo = ['WIND, HOURLY SPEED (m/s)',
              'AIR RELATIVE HUMIDITY, HOURLY (%)',
              'DEW POINT TEMPERATURE (°C)',
              'AIR TEMPERATURE - DRY BULB, HOURLY (°C)',
             'ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)']


    y = dados['GLOBAL RADIATION (Kj/m²)']
    X = dados[colunas_araujo]

    size = 0.6
    train_size = int(len(y)*size)

    X_treino, X_teste = X[:train_size], X[train_size:]
    y_treino, y_teste = y[:train_size], y[train_size:]

    val_size = 0.5
    val = int(len(y_teste) * val_size)

    x_val, y_val     = X_teste.iloc[val:], y_teste.iloc[val:]
    X_teste, y_teste = X_teste.iloc[:val], y_teste.iloc[:val]

   # print(f'Dados carregados, limpos e divididos! /n dados treino {X_treino.shape} dados teste {X_teste.shape}')
    return X_treino, X_teste, y_treino, y_teste, x_val, y_val, dados 

def remove_outliers(df): 
    outliers_by_hour = {}
    df['hour'] = df.index.hour
    for hour in range(24):
        hour_data = df[df['hour'] == hour]['GLOBAL RADIATION (Kj/m²)']
        Q1 = hour_data.quantile(0.25)
        Q3 = hour_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = hour_data[(hour_data < lower) | (hour_data > upper)]
        outliers_by_hour[hour] = outliers
        df = df[~df.index.isin(outliers.index)]
    return df
        
def main(path):
    df_quebrado = []
    
    df = pd.read_parquet('dados/dadosLimpos.parquet')
    #   X_treino, X_teste, y_treino, y_teste, x_val, y_val, df =  preprocessar_dados(path)

    periodos = {
    'verao':     df[df.index.month.isin([12, 1, 2])],
    'outono':    df[df.index.month.isin([3, 4, 5])],
    'inverno':   df[df.index.month.isin([6, 7, 8])],
    'primavera': df[df.index.month.isin([9, 10, 11])]
}
    
    for k in periodos:
        dados = periodos[k]
        X_treino, X_teste, y_treino, y_teste, x_val, y_val, df =  preprocessar_dados(dados)
        scaler = MinMaxScaler()
        X_treino_escalado = scaler.fit_transform(X_treino)
        X_teste_escalado = scaler.transform(X_teste)
        x_val = scaler.transform(x_val)
        
        scaler_y = MinMaxScaler()
        y_treino_escalado = scaler_y.fit_transform(y_treino.to_numpy().reshape(-1,1)).ravel()
        y_val = scaler_y.transform(y_val.to_numpy().reshape(-1,1)).ravel()
        
        model = RNALM(input_dim=X_treino.shape[1])
        model.fit(X_treino_escalado, y_treino_escalado, x_val, y_val)
        
        predicao = model.predict(X_teste_escalado)
        predicao = scaler_y.inverse_transform(predicao.reshape(-1,1)).ravel()

        metricas = calcular_metricas(y_test = y_teste, y_pred = predicao)

        xgb = XGBRegressor()
       # xgb.fit(X_treino, y_treino)
        #xgb_predict = xgb.predict(X_teste)
        #xgb_metricas = calcular_metricas(y_teste, xgb_predict)
        

# Parameter grid for RandomizedSearchCV with your specified hyperparameters
        param_grid = {
    # Maximum depth of a tree. Samples integers between 3 and 10.
            'max_depth': randint(3, 11),

    # Minimum sum of instance weight needed in a child. Samples integers between 1 and 10.
            'min_child_weight': randint(1, 11),

    # Minimum loss reduction required to make a further partition. Samples floats between 0.0 and 0.5.
            'gamma': uniform(0, 0.5),

    # Maximum number of categories for one-hot encoding. Samples integers between 5 and 30.
            'max_cat_threshold': randint(5, 31)
        }

# --- How to use it ---
# from sklearn.model_selection import RandomizedSearchCV
# from xgboost import XGBClassifier
#
# # 1. Create the model instance
# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#
# # 2. Create the RandomizedSearchCV object
# # n_iter controls how many different combinations to try
        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid,
            n_iter=1000,       # Try 50 different combinations
            cv=10,            # Use 5-fold cross-validation
            verbose=2,
            random_state=42,
            n_jobs=-1        # Use all available CPU cores
        )
#
# # 3. Fit the random search to your data
        random_search.fit(X_treino_escalado, y_treino_escalado)

        predic = random_search.predict(x_val)
        predic = predic.reshape(-1,1).ravel()
        xgb_metricas = calcular_metricas(y_val, predic)

# # 4. Check the best parameters found
# # print("Best parameters found: ", random_search.best_params_)
        
        print(k)
        print(metricas)
        print('XGB', xgb_metricas)
    #plt.figure(figsize=(12, 8))
    #plot_tree(model, filled=True, rounded=True, feature_names=X_treino.columns)
    #plt.show()



    


  #      X_treino, X_teste, y_treino, y_teste, _, __ ,___ = preprocessar_
        
        
'''
    scaler = StandardScaler()
    X_treino_escalado = scaler.fit_transform(X_treino)
    X_teste_escalado = scaler.transform(X_teste)
    x_val = scaler.transform(x_val)

    scaler_y = StandardScaler()
    y_treino_escalado = scaler_y.fit_transform(y_treino.to_numpy().reshape(-1,1)).ravel()
    y_val = scaler_y.transform(y_val.to_numpy().reshape(-1,1)).ravel()

    model.fit(X_treino_escalado, y_treino_escalado)
    predicao = model.predict(X_teste)
    predicao = scaler_y.inverse_transform(predicao.reshape(-1,1)).ravel()

    metricas = calcular_metricas(y_test = y_teste, y_pred = predicao)
    print(metricas)

    #plt.figure(figsize=(12, 8))
    #plot_tree(model, filled=True, rounded=True, feature_names=X_treino.columns)
    #plt.show()

    return 
'''
if __name__ == "__main__":
    main(path)




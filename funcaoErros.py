#funcaoErros.py

import numpy as np

def calcular_metricas(y_test, y_pred):
    y_test = np.array(y_test).ravel()
    y_pred = np.array(y_pred).ravel()
    
    mse = np.mean((y_test - y_pred)**2)
    
    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(y_test - y_pred))
    
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test==0, 1e-8, y_test))) * 100
    
    ss_res = np.sum((y_test - y_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    
    r2 = 1 - ss_res / ss_tot

    evs = 1 - np.var(y_test - y_pred) / np.var(y_test)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'EVS': evs
    }

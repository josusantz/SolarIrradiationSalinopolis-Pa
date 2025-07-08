import pandas as pd
# import numpy as np

path = "data/dados_A215_H_2008-06-13_2024-01-01.csv"

df = pd.read_csv(path, header = 9, sep =';')
df.drop(columns= ['Unnamed: 22'], axis =1,inplace = True)
df.replace(to_replace = ',', value = '.', regex = True, inplace =True)
df = df.apply(pd.to_numeric, errors = 'ignore')
df['Hora Medicao'] = df['Hora Medicao']/100
df['Hora Medicao'] = df['Hora Medicao'].astype(int).astype(str).str.zfill(2)
df['Hora Medicao'] = pd.to_datetime(df['Hora Medicao'], format='%H').dt.time
df['Data_Hora'] = pd.to_datetime(df['Data Medicao'] +' '+ df['Hora Medicao'].astype(str))
df.set_index('Data_Hora', inplace= True)
# df.dropna(inplace = True)
df['hour']= df.index.hour
# df = df[(df.index.hour >= 12) & (df.index.hour <=20)]
df.drop(columns=['Data Medicao', 'Hora Medicao', 'hour'], inplace = True)
df.sort_index(inplace =True)
df.fillna(0, inplace =True)
# df['SMA_5'] = df['RADIACAO GLOBAL(Kj/m²)'].rolling(window=5,center =True).mean()
# df['VEL_VENTO_km_h'] = df['VENTO, VELOCIDADE HORARIA(m/s)'] * 3.6
# df['ST'] = 33 + (10 * np.sqrt(df['VEL_VENTO_km_h']) + 10.45 - df['VEL_VENTO_km_h']) * (df['TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)'] - 33) / 22
# df.dropna(inplace =True)
# df.drop(columns=['ST', 'SMA_5', 'VEL_VENTO_km_h'], inplace = True)

columns_translation = {
    'PRECIPITACAO TOTAL, HORARIO(mm)': 'TOTAL PRECIPITATION, HOURLY (mm)',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB)': 'ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)',
    'PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB)': 'ATMOSPHERIC PRESSURE REDUCED TO SEA LEVEL, AUT (mB)',
    'PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT)(mB)': 'ATMOSPHERIC PRESSURE MAX. IN THE PREVIOUS HOUR (AUT) (mB)',
    'PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT)(mB)': 'ATMOSPHERIC PRESSURE MIN. IN THE PREVIOUS HOUR (AUT) (mB)',
    'RADIACAO GLOBAL(Kj/m²)': 'GLOBAL RADIATION (Kj/m²)',
    'TEMPERATURA DA CPU DA ESTACAO(°C)': 'STATION CPU TEMPERATURE (°C)',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)': 'AIR TEMPERATURE - DRY BULB, HOURLY (°C)',
    'TEMPERATURA DO PONTO DE ORVALHO(°C)': 'DEW POINT TEMPERATURE (°C)',
    'TEMPERATURA MAXIMA NA HORA ANT. (AUT)(°C)': 'MAXIMUM TEMPERATURE IN THE PREVIOUS HOUR (AUT) (°C)',
    'TEMPERATURA MINIMA NA HORA ANT. (AUT)(°C)': 'MINIMUM TEMPERATURE IN THE PREVIOUS HOUR (AUT) (°C)',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT)(°C)': 'MAXIMUM DEW POINT TEMPERATURE IN THE PREVIOUS HOUR (AUT) (°C)',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT)(°C)': 'MINIMUM DEW POINT TEMPERATURE IN THE PREVIOUS HOUR (AUT) (°C)',
    'TENSAO DA BATERIA DA ESTACAO(V)': 'STATION BATTERY VOLTAGE (V)',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT)(%)': 'RELATIVE HUMIDITY MAX. IN THE PREVIOUS HOUR (AUT) (%)',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT)(%)': 'RELATIVE HUMIDITY MIN. IN THE PREVIOUS HOUR (AUT) (%)',
    'UMIDADE RELATIVA DO AR, HORARIA(%)': 'AIR RELATIVE HUMIDITY, HOURLY (%)',
    'VENTO, DIRECAO HORARIA (gr)(° (gr))': 'WIND, HOURLY DIRECTION (gr) (° (gr))',
    'VENTO, RAJADA MAXIMA(m/s)': 'WIND, MAXIMUM GUST (m/s)',
    'VENTO, VELOCIDADE HORARIA(m/s)': 'WIND, HOURLY SPEED (m/s)'
}

df.rename(columns=columns_translation, inplace=True)
print(df.columns)
df.to_parquet('data/processeddata.parquet')
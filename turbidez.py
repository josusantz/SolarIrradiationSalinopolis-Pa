import pandas as pd
import pvlib
import pytz
import numpy as np

# --- 0. Informações da sua estação (Salinópolis, Pará, Brasil) ---
latitude_estacao =  -0.61888888
longitude_estacao = -47.35666666
altitude_estacao = 23.18 # em metros

data_path ="data\processeddata.parquet"

df_seu_dado = pd.read_parquet(data_path)

cpu_columns = ['STATION CPU TEMPERATURE (°C)',
                'STATION BATTERY VOLTAGE (V)']

df_seu_dado.drop(columns=cpu_columns, inplace=True)


tz = 'America/Belem'


# A coluna de tempo para pvlib.clearsky.ineichen será o índice do DataFrame se estiver localizado.
time_points = df_seu_dado.index

# --- 1. Estimar o Vapor d'água (Precipitable Water - pw) a partir de Umidade e Temperatura ---
# Renomeie as colunas temporariamente para facilitar o uso na função pvlib
temp_air_celsius = df_seu_dado['AIR TEMPERATURE - DRY BULB, HOURLY (°C)']
relative_humidity_percent = df_seu_dado['AIR RELATIVE HUMIDITY, HOURLY (%)']
pressure_mB = df_seu_dado['ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)']

# Converta a pressão de mB para Pascal (1 mB = 100 Pa), pois a função pvlib espera Pascal.
pressure_pa = pressure_mB * 100

df_seu_dado['precipitable_water'] = pvlib.atmosphere.gueymard94_pw(
    temp_air=temp_air_celsius,
    relative_humidity=relative_humidity_percent,
    pressure=pressure_pa
)

print("Vapor d'água estimado (precipitable_water):")
print(df_seu_dado[['AIR TEMPERATURE - DRY BULB, HOURLY (°C)',
                   'AIR RELATIVE HUMIDITY, HOURLY (%)',
                   'ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)',
                   'precipitable_water']].head())

# --- 2. Definir o AOD (Profundidade Óptica de Aerossóis) ---
# Usaremos o valor de 0.2 como ponto de partida, conforme a discussão anterior.
aod_550nm = 0.2

# --- 3. Calcular a Turbidez de Linke (TL_2) usando a fórmula de Ineichen ---
# Pressão padrão ao nível do mar (em mB para consistência com seus dados)
p0_mB = 1013.25

# Razão de pressão P0/P
pressure_ratio = p0_mB / pressure_mB

# O termo ln(w) precisa lidar com casos onde w pode ser muito próximo de zero.
# Substitui 0 por um valor pequeno e trata NaNs.
ln_w = np.log(df_seu_dado['precipitable_water'].replace(0, np.nan).fillna(1e-6))

df_seu_dado['linke_turbidity'] = (
    3.91 * np.exp(0.689 * pressure_ratio) * aod_550nm +
    0.376 * ln_w +
    2 +
    0.54 * pressure_ratio -
    0.5 * (pressure_ratio**2) +
    0.16 * (pressure_ratio**3)
)

print("\nTurbidez de Linke estimada (linke_turbidity):")
print(df_seu_dado[['precipitable_water', 'linke_turbidity']].head())

# --- 4. (Próximo passo após este código) Calcular o GHI de Céu Limpo (GHIcs) ---
# Agora que você tem 'linke_turbidity' em seu DataFrame,
# pode usar esta coluna como input para pvlib.clearsky.ineichen.
# Exemplo de como você faria isso (NÃO execute sem o resto do DataFrame):
# df_seu_dado['GHIcs_Ineichen'] = pvlib.clearsky.ineichen(
#     time=time_points,
#     latitude=latitude_estacao,
#     longitude=longitude_estacao,
#     altitude=altitude_estacao,
#     linke_turbidity=df_seu_dado['linke_turbidity'] # Usando a coluna calculada
# )
import pandas as pd
import pvlib
import pytz
import numpy as np

# --- 0. Informações da sua estação (Salinópolis, Pará, Brasil) ---
latitude_estacao = -0.61888888
longitude_estacao = -47.35666666
altitude_estacao = 23.18 # em metros

# --- Carregue seus dados reais ---
path = "data/processeddata.parquet"
df_seu_dado = pd.read_parquet(path)

# --- Localizar ou Converter o Fuso Horário do Index ---
fuso_horario_local = 'America/Belem'

if df_seu_dado.index.tz is None:
    df_seu_dado.index = df_seu_dado.index.tz_localize(fuso_horario_local, ambiguous='NaT', nonexistent='NaT')
    print("\nFuso horário localizado para:", fuso_horario_local)
elif str(df_seu_dado.index.tz) != fuso_horario_local:
    df_seu_dado.index = df_seu_dado.index.tz_convert(fuso_horario_local)
    print("\nFuso horário convertido para:", fuso_horario_local)
else:
    print("\nFuso horário já é", fuso_horario_local, "(nenhuma ação necessária).")

print("Tipo de dado do índice após ajuste:", df_seu_dado.index.dtype)
print("Primeiras linhas do DataFrame com índice ajustado:")
print(df_seu_dado.head())

# --- TRATAMENTO DE NA_N NAS COLUNAS DE ENTRADA DO MODELO ---
# É CRÍTICO garantir que essas colunas não tenham NaNs antes de calcular precipitable_water
# Interpolar as colunas de entrada antes de calcular precipitable_water
for col in ['AIR TEMPERATURE - DRY BULB, HOURLY (°C)', 'AIR RELATIVE HUMIDITY, HOURLY (%)', 'ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)']:
    if df_seu_dado[col].isnull().any():
        print(f"\nAVISO: NaNs encontrados na coluna '{col}'. Interpolando...")
        df_seu_dado[col] = df_seu_dado[col].interpolate(method='linear', limit_direction='both', limit_area='inside')
        # Se ainda houver NaNs (ex: no início/fim), preencher com a média ou remover as linhas.
        if df_seu_dado[col].isnull().any():
            print(f"AVISO: Ainda há NaNs em '{col}' após interpolação. Preenchendo com a média.")
            df_seu_dado[col].fillna(df_seu_dado[col].mean(), inplace=True)

# --- 1. Estimar o Vapor d'água (Precipitable Water - pw) a partir de Umidade e Temperatura ---
temp_air_celsius = df_seu_dado['AIR TEMPERATURE - DRY BULB, HOURLY (°C)']
relative_humidity_percent = df_seu_dado['AIR RELATIVE HUMIDITY, HOURLY (%)']
pressure_mB = df_seu_dado['ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)']

df_seu_dado['precipitable_water'] = pvlib.atmosphere.gueymard94_pw(
    temp_air=temp_air_celsius,
    relative_humidity=relative_humidity_percent
)

print("\nVapor d'água estimado (precipitable_water):")
print(df_seu_dado[['AIR TEMPERATURE - DRY BULB, HOURLY (°C)',
                   'AIR RELATIVE HUMIDITY, HOURLY (%)',
                   'ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)',
                   'precipitable_water']].head())

# --- TRATAMENTO DE NA_N NA 'precipitable_water' ---
# Este tratamento é redundante se as colunas de entrada já foram limpas, mas é uma boa prática
# para pegar NaNs que possam ter surgido por outros motivos (e.g., divisão por zero etc.).
df_seu_dado['precipitable_water'] = df_seu_dado['precipitable_water'].interpolate(method='linear', limit_direction='both', limit_area='inside')
if df_seu_dado['precipitable_water'].isnull().any():
    print("\nAVISO: Ainda há NaNs em 'precipitable_water' após a interpolação. Preenchendo com a média.")
    df_seu_dado['precipitable_water'].fillna(df_seu_dado['precipitable_water'].mean(), inplace=True)


# --- 2. Definir o AOD (Profundidade Óptica de Aerossóis) ---
aod_550nm = 0.2

# --- 3. Calcular a Turbidez de Linke (TL_2) usando a fórmula de Ineichen ---
p0_mB = 1013.25
pressure_ratio = p0_mB / pressure_mB

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

# --- TRATAMENTO DE NA_N NA 'linke_turbidity' ---
df_seu_dado['linke_turbidity'] = df_seu_dado['linke_turbidity'].fillna(df_seu_dado['linke_turbidity'].mean())

# Salva o DataFrame com as novas colunas
df_seu_dado.to_parquet('data/linke_turbidity.parquet')
print("\nDataFrame salvo em 'data/linke_turbidity.parquet' com as novas colunas 'precipitable_water' e 'linke_turbidity'.")

solpos = pvlib.solarposition.get_solarposition(df_seu_dado.index,latitude_estacao,longitude_estacao, altitude=altitude_estacao)

df_seu_dado["apparent_zenith"] = solpos["apparent_zenith"]

df_seu_dado["airmass_relative"] = pvlib.atmosphere.get_relative_airmass(df_seu_dado["apparent_zenith"])

df_seu_dado["airmass_absolute"] = pvlib.atmosphere.get_absolute_airmass(df_seu_dado["airmass_relative"],
                                                                         pressure=df_seu_dado['ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)'] * 100)

# --- 4. Calcular o GHI de Céu Limpo (GHIcs) utilizando o modelo Ineichen ---
clearsky = pvlib.clearsky.ineichen(
    apparent_zenith = df_seu_dado["apparent_zenith"],
    airmass_absolute = df_seu_dado["airmass_absolute"],
    altitude=altitude_estacao,
    linke_turbidity=df_seu_dado['linke_turbidity']
)

for col in clearsky.columns:
    df_seu_dado[col] = clearsky[col]

print("\nGHI de Céu Limpo (GHIcs) calculado pelo modelo Ineichen:")
print(df_seu_dado[['precipitable_water', 'linke_turbidity']].head())

df_seu_dado['GHI_measured_Wm2_plot'] = df_seu_dado['GLOBAL RADIATION (Kj/m²)'] / 3.6
df_seu_dado.loc[df_seu_dado['GHI_measured_Wm2_plot'] < 0, 'GHI_measured_Wm2_plot'] = 0

# Salva o DataFrame novamente com a nova coluna GHIcs_Ineichen
df_seu_dado.to_parquet('data/ghi_cs_calculado.parquet')
print("\nDataFrame final salvo em 'data/ghi_cs_calculado.parquet' com a coluna 'GHIcs_Ineichen'.")
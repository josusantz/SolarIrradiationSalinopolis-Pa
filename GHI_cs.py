import pandas as pd
import pvlib
import pytz # Importado para referência, mas pvlib lida com fusos.
import numpy as np
import matplotlib.pyplot as plt # Para plotagem de verificação
import matplotlib.dates as mdates # Para formatar datas no eixo x

# --- INÍCIO DA PRIMEIRA PARTE DO SEU SCRIPT (Leitura do CSV e salvamento em Parquet) ---
# Esta parte parece estar correta em criar um DatetimeIndex naive que representa UTC.
path_csv = "data/dados_A215_H_2008-06-13_2024-01-01.csv"

df_original = pd.read_csv(path_csv, header=9, sep=';')
df_original.drop(columns=['Unnamed: 22'], axis=1, inplace=True)
df_original.replace(to_replace=',', value='.', regex=True, inplace=True)
df_original = df_original.apply(pd.to_numeric, errors='ignore')

# Tratamento da Hora Medicao
df_original['Hora Medicao'] = df_original['Hora Medicao'] / 100
df_original['Hora Medicao'] = df_original['Hora Medicao'].astype(int).astype(str).str.zfill(2)
df_original['Hora Medicao'] = pd.to_datetime(df_original['Hora Medicao'], format='%H').dt.time

# Criar Data_Hora (este índice será naive, representando UTC)
df_original['Data_Hora'] = pd.to_datetime(df_original['Data Medicao'] + ' ' + df_original['Hora Medicao'].astype(str))
df_original.set_index('Data_Hora', inplace=True)

df_original.drop(columns=['Data Medicao', 'Hora Medicao'], inplace=True) # Removido 'hour' pois não era usado posteriormente
df_original.sort_index(inplace=True)
df_original.fillna(0, inplace=True) # Preencher NaNs com 0 conforme seu script original

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

df_original.rename(columns=columns_translation, inplace=True)
print("Colunas do DataFrame original após renomear:")
print(df_original.columns)
df_original.to_parquet('data/processeddata_utc_naive.parquet') # Salvar com nome que indica ser UTC naive
print("DataFrame original processado e salvo em 'data/processeddata_utc_naive.parquet'")
# --- FIM DA PRIMEIRA PARTE DO SEU SCRIPT ---


# --- INÍCIO DA SEGUNDA PARTE DO SEU SCRIPT (Cálculos pvlib com fuso horário corrigido) ---
# --- 0. Informações da sua estação (Salinópolis, Pará, Brasil) ---
latitude_estacao = -0.61888888
longitude_estacao = -47.35666666
altitude_estacao = 23.18 # em metros
fuso_horario_local_str = 'America/Belem' # String para o fuso local

# --- Carregamento e Preparação Inicial dos Dados ---
path_parquet = "data/processeddata_utc_naive.parquet" # Carregar o arquivo salvo
df_seu_dado = pd.read_parquet(path_parquet)
print(f"\nDataFrame carregado de '{path_parquet}'. Primeiras linhas do índice:")

# --- CORREÇÃO DA LOCALIZAÇÃO DO FUSO HORÁRIO ---
if df_seu_dado.index.tzinfo is None:
    print("Índice é naive. Assumindo que é UTC e convertendo para America/Belem.")
    # 1. Localizar para UTC (pois os dados originais são UTC)
    # 2. Converter para o fuso horário local de Salinópolis
    df_seu_dado.index = df_seu_dado.index.tz_localize('UTC').tz_convert(fuso_horario_local_str)
    print("Índice agora localizado para UTC e convertido para America/Belem.")
elif df_seu_dado.index.tzinfo.zone != fuso_horario_local_str:
    print(f"Índice já está localizado para {df_seu_dado.index.tzinfo.zone}. Convertendo para {fuso_horario_local_str}.")
    df_seu_dado.index = df_seu_dado.index.tz_convert(fuso_horario_local_str)
else:
    print(f"Índice já está corretamente localizado para {fuso_horario_local_str}.")

print("Primeiras linhas do índice após ajuste de fuso horário:")


# --- Preparar Variáveis de Entrada (com tratamento de NaNs se necessário) ---
cols_to_interpolate_input = [
    'AIR TEMPERATURE - DRY BULB, HOURLY (°C)',
    'AIR RELATIVE HUMIDITY, HOURLY (%)',
    'ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)'
]
for col in cols_to_interpolate_input:
    if col in df_seu_dado.columns:
        df_seu_dado[col] = df_seu_dado[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    else:
        print(f"Aviso: Coluna de input {col} não encontrada para interpolação.")


temp_air_celsius = df_seu_dado['AIR TEMPERATURE - DRY BULB, HOURLY (°C)']
relative_humidity_percent = df_seu_dado['AIR RELATIVE HUMIDITY, HOURLY (%)']
pressure_mB = df_seu_dado['ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)']

# --- 1. Estimar o Vapor d'água (Precipitable Water - pw) ---
df_seu_dado['precipitable_water'] = pvlib.atmosphere.gueymard94_pw(
    temp_air=temp_air_celsius,
    relative_humidity=relative_humidity_percent
)
df_seu_dado['precipitable_water'] = df_seu_dado['precipitable_water'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# --- 2. Definir o AOD (Profundidade Óptica de Aerossóis) ---
aod_550nm = 0.2

# --- 3. Calcular a Turbidez de Linke (TL) ---
p0_mB = 1013.25
pressure_ratio = p0_mB / pressure_mB
ln_w = np.log(df_seu_dado['precipitable_water'].replace(0, 1e-6).fillna(1e-6))
df_seu_dado['linke_turbidity'] = (
    3.91 * np.exp(0.689 * pressure_ratio) * aod_550nm +
    0.376 * ln_w +
    2 +
    0.54 * pressure_ratio -
    0.5 * (pressure_ratio**2) +
    0.16 * (pressure_ratio**3)
)
print("\nTurbidez de Linke estimada (linke_turbidity) antes da interpolação final:")
print(df_seu_dado[['precipitable_water', 'linke_turbidity']].head())
print(f"NaNs em linke_turbidity antes da interpolação: {df_seu_dado['linke_turbidity'].isnull().sum()}")

# --- 4. Preparar Inputs para o Modelo de Céu Limpo ---
df_seu_dado['pressure_pascals'] = pressure_mB * 100

solar_position = pvlib.solarposition.get_solarposition(
    time=df_seu_dado.index,
    latitude=latitude_estacao,
    longitude=longitude_estacao,
    altitude=altitude_estacao,
    temperature=temp_air_celsius,
    pressure=df_seu_dado['pressure_pascals']
)
df_seu_dado['apparent_zenith'] = solar_position['apparent_zenith']
df_seu_dado['solar_azimuth'] = solar_position['azimuth']
df_seu_dado['dni_extra'] = pvlib.irradiance.get_extra_radiation(df_seu_dado.index)
airmass_relative = pvlib.atmosphere.get_relative_airmass(zenith=df_seu_dado['apparent_zenith'])
df_seu_dado['airmass_absolute'] = pvlib.atmosphere.get_absolute_airmass(
    airmass_relative=airmass_relative,
    pressure=df_seu_dado['pressure_pascals']
)

cols_to_interpolate_final = ['linke_turbidity', 'apparent_zenith', 'airmass_absolute', 'dni_extra']
for col in cols_to_interpolate_final:
    if col in df_seu_dado.columns:
        df_seu_dado[col] = df_seu_dado[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    else:
        print(f"Aviso: Coluna {col} não foi encontrada para interpolação final.")

if df_seu_dado['linke_turbidity'].isnull().any():
    print("Aviso: 'linke_turbidity' ainda contém NaNs. Preenchendo com valor default 3.")
    df_seu_dado['linke_turbidity'].fillna(3, inplace=True)

# --- 5. Calcular o GHI de Céu Limpo (GHIcs) com Ineichen ---
clearsky_output = pvlib.clearsky.ineichen(
    apparent_zenith=df_seu_dado['apparent_zenith'],
    airmass_absolute=df_seu_dado['airmass_absolute'],
    linke_turbidity=df_seu_dado['linke_turbidity'],
    altitude=altitude_estacao,
    dni_extra=df_seu_dado['dni_extra']
)

for col in clearsky_output.columns:
    df_seu_dado[col] = clearsky_output[col]

print("\nCabeçalho do DataFrame com GHIcs_Ineichen:")
print(df_seu_dado[['ghi', 'linke_turbidity', 'apparent_zenith', 'airmass_absolute', 'dni_extra']].head())
print(f"\nNaNs em GHIcs_Ineichen: {df_seu_dado['ghi'].isnull().sum()}")
print(f"Valores de GHIcs_Ineichen durante a noite (zênite > 90 e GHIcs != 0): {len(df_seu_dado[(df_seu_dado['apparent_zenith'] > 90) & (df_seu_dado['ghi'] != 0)])}")
print(f"Valores de GHIcs_Ineichen durante a noite (zênite > 90 e GHIcs é NaN): {len(df_seu_dado[(df_seu_dado['apparent_zenith'] > 90) & (df_seu_dado['ghi'].isnull())])}")

# --- 6. Salvar os Resultados ---
df_seu_dado['GHI_measured_Wm2_plot'] = df_seu_dado['GLOBAL RADIATION (Kj/m²)'] / 3.6
df_seu_dado.loc[df_seu_dado['GHI_measured_Wm2_plot'] < 0, 'GHI_measured_Wm2_plot'] = 0
df_seu_dado.rename(columns={'ghi': 'GHIcs_Ineichen'}, inplace=True)

output_path_final = 'data/Salinopolis_GHI_ClearSky_Calculado_FusoCorrigido.parquet'
df_seu_dado.to_parquet(output_path_final)
print(f"\nDataFrame com GHI de céu limpo (fuso corrigido) salvo em: {output_path_final}")

# --- 7. Plotar GHI vs. Ângulo de Zênite para Verificação ---
df_plot = pd.read_parquet(output_path_final) 

if 'GLOBAL RADIATION (Kj/m²)' in df_plot.columns:
    df_plot['GHI_measured_Wm2_plot'] = df_plot['GLOBAL RADIATION (Kj/m²)'] / 3.6
    df_plot.loc[df_plot['GHI_measured_Wm2_plot'] < 0, 'GHI_measured_Wm2_plot'] = 0

    fig, ax_scatter = plt.subplots(figsize=(12, 7))
    ax_scatter.scatter(df_plot['apparent_zenith'], df_plot['GHI_measured_Wm2_plot'],
                       label='GHI Medido (W/m²)', color='blue', alpha=0.2, s=5)
    if 'GHIcs_Ineichen' in df_plot.columns:
        ax_scatter.scatter(df_plot['apparent_zenith'], df_plot['GHIcs_Ineichen'],
                           label='GHI Céu Limpo (W/m²)', color='red', alpha=0.2, s=5)
    ax_scatter.set_xlabel("Ângulo de Zênite Aparente (graus)")
    ax_scatter.set_ylabel("Irradiação Solar Global Horizontal (W/m²)")
    ax_scatter.set_title("Irradiação Solar vs. Ângulo de Zênite (Após Correção de Fuso)")
    ax_scatter.set_xlim(0, 180)
    ax_scatter.set_ylim(bottom=-50)
    ax_scatter.axvline(90, color='gray', linestyle=':', label='Zênite 90° (Horizonte)')
    ax_scatter.grid(True, linestyle=':', alpha=0.7)
    ax_scatter.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Coluna 'GLOBAL RADIATION (Kj/m²)' não encontrada para plotagem de verificação.")


# --- 8. Aplicar Filtro de Zênite e Calcular kt ---
# Carregar o DataFrame com os cálculos de céu limpo e fuso corrigido
df_para_analise = pd.read_parquet(output_path_final)
# Certificar que o índice está no fuso horário correto (deve estar se foi salvo corretamente)
if df_para_analise.index.tzinfo is None:
    df_para_analise.index = df_para_analise.index.tz_localize('UTC').tz_convert(fuso_horario_local_str)
elif df_para_analise.index.tzinfo.zone != fuso_horario_local_str:
    df_para_analise.index = df_para_analise.index.tz_convert(fuso_horario_local_str)

print(f"\nNúmero de registros antes do filtro de zênite (alvorada/crepúsculo): {len(df_para_analise)}")

# Definir o limiar de zênite para remover alvorada/crepúsculo
# Valores menores que este serão mantidos (sol mais alto)
zenith_threshold_day = 85.0 
daylight_filter_strict = df_para_analise['apparent_zenith'] < zenith_threshold_day
df_filtrado_dia = df_para_analise[daylight_filter_strict].copy() # Usar .copy() para evitar SettingWithCopyWarning

print(f"Número de registros após filtro de zênite < {zenith_threshold_day}°: {len(df_filtrado_dia)}")

# Prosseguir com o cálculo de kt usando o df_filtrado_dia
df_filtrado_dia['GHI_measured_Wm2'] = df_filtrado_dia['GLOBAL RADIATION (Kj/m²)'] / 3.6
df_filtrado_dia.loc[df_filtrado_dia['GHI_measured_Wm2'] < 0, 'GHI_measured_Wm2'] = 0

ghi_cs_min_threshold = 1 # W/m²
df_filtrado_dia['kt'] = np.nan

# Calcular kt apenas onde GHIcs_Ineichen é significativo (e agora, apenas para os dados já filtrados pelo zênite)
condition_daytime_kt = df_filtrado_dia['GHIcs_Ineichen'] >= ghi_cs_min_threshold
df_filtrado_dia.loc[condition_daytime_kt, 'kt'] = \
    df_filtrado_dia['GHI_measured_Wm2'][condition_daytime_kt] / df_filtrado_dia['GHIcs_Ineichen'][condition_daytime_kt]

# Tratamento de Valores de kt
kt_max_cap = 1.2 # Exemplo, ajuste conforme necessário
df_filtrado_dia.loc[df_filtrado_dia['kt'] > kt_max_cap, 'kt'] = kt_max_cap
df_filtrado_dia.loc[df_filtrado_dia['kt'] < 0, 'kt'] = 0 # Deve ser redundante se GHI_measured_Wm2 >=0
# Preencher NaNs restantes em 'kt' (se GHIcs_Ineichen < ghi_cs_min_threshold mesmo após filtro de zênite)
df_filtrado_dia['kt'].fillna(0, inplace=True) 

print("\nDataFrame (filtrado por zênite) com kt calculado:")
print(df_filtrado_dia[['GLOBAL RADIATION (Kj/m²)', 'GHI_measured_Wm2', 'GHIcs_Ineichen', 'apparent_zenith', 'kt']].head())
print(f"\nDescrição estatística de kt (dados filtrados por zênite < {zenith_threshold_day}°):")
print(df_filtrado_dia['kt'].describe())

# Salvar o DataFrame final que contém kt e está filtrado
output_path_kt_filtrado = f'data/Salinopolis_kt_Calculado_FusoCorrigido_Zenith{int(zenith_threshold_day)}.parquet'
df_filtrado_dia.to_parquet(output_path_kt_filtrado)
print(f"\nDataFrame com kt (fuso corrigido e zênite < {zenith_threshold_day}°) salvo em: {output_path_kt_filtrado}")

# Plotar o GHI vs Zênite para o df_filtrado_dia para ver o efeito do filtro
if 'GLOBAL RADIATION (Kj/m²)' in df_filtrado_dia.columns:
    fig, ax_scatter_filtrado = plt.subplots(figsize=(12, 7))
    ax_scatter_filtrado.scatter(df_filtrado_dia['apparent_zenith'], df_filtrado_dia['GHI_measured_Wm2'],
                       label='GHI Medido (W/m²) - Filtrado', color='green', alpha=0.3, s=5)
    if 'GHIcs_Ineichen' in df_filtrado_dia.columns:
        ax_scatter_filtrado.scatter(df_filtrado_dia['apparent_zenith'], df_filtrado_dia['GHIcs_Ineichen'],
                           label='GHI Céu Limpo (W/m²) - Filtrado', color='orange', alpha=0.3, s=5)
    ax_scatter_filtrado.set_xlabel("Ângulo de Zênite Aparente (graus)")
    ax_scatter_filtrado.set_ylabel("Irradiação Solar Global Horizontal (W/m²)")
    ax_scatter_filtrado.set_title(f"Irradiação Solar vs. Zênite (Zênite < {zenith_threshold_day}°)")
    ax_scatter_filtrado.set_xlim(0, zenith_threshold_day + 5) # Ajustar xlim para o intervalo filtrado
    ax_scatter_filtrado.set_ylim(bottom=-50)
    ax_scatter_filtrado.grid(True, linestyle=':', alpha=0.7)
    ax_scatter_filtrado.legend()
    plt.tight_layout()
    plt.show()

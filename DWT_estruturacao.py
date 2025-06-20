import pandas as pd
import numpy as np
import pywt # pip install PyWavelets
import matplotlib.pyplot as plt
import os
from pywt import wavedec


print("Iniciando o script DWT_estruturado.py" \
"")
# --- 1. CONFIGURAÇÃO E CARREGAMENTO DE DADOS ---
# Parâmetros para sua revisão
WAVELET_TYPE = 'db4'
DECOMPOSITION_LEVEL = 4
N_LAGS = 10

# Caminho para o arquivo final da etapa anterior (filtrado por zênite)
path_zenith_filtered = f'data/Salinopolis_kt_Calculado_FusoCorrigido_Zenith{int(85)}.parquet'
# Caminho para o arquivo com fuso corrigido, mas *antes* do filtro de zênite (necessário para DWT contínua)
path_fuso_corrigido = 'data/Salinopolis_GHI_ClearSky_Calculado_FusoCorrigido.parquet'

if not os.path.exists(path_fuso_corrigido):
    raise FileNotFoundError(f"Arquivo necessário não encontrado: {path_fuso_corrigido}")
if not os.path.exists(path_zenith_filtered):
    raise FileNotFoundError(f"Arquivo necessário não encontrado: {path_zenith_filtered}")

df_completo = pd.read_parquet(path_fuso_corrigido)
df_filtrado = pd.read_parquet(path_zenith_filtered)

print(f"Dados completos carregados: {len(df_completo)} registros.")
print(f"Dados filtrados por zênite carregados: {len(df_filtrado)} registros.")


# --- 2. APLICAÇÃO DA DWT NA SÉRIE TEMPORAL CONTÍNUA ---
# Usar a série GHI medida (em W/m²) do DataFrame completo para garantir continuidade.
ghi_series_continua = df_completo['GLOBAL RADIATION (Kj/m²)'] / 3.6
ghi_series_continua.loc[ghi_series_continua < 0] = 0
ghi_series_continua.interpolate(method='linear', inplace=True)
ghi_series_continua = ghi_series_continua.bfill()
ghi_series_continua = ghi_series_continua.ffill()

# Decompor a série com DWT
coeffs = wavedec(ghi_series_continua, wavelet=WAVELET_TYPE, level=DECOMPOSITION_LEVEL)

# Função para reconstruir cada componente DWT com o mesmo tamanho da série original
def reconstruct_component(coeffs, wavelet, level, coef_type, orig_len):
    coeffs_mod = [np.zeros_like(c) for c in coeffs]
    if coef_type == 'a':
        coeffs_mod[0] = coeffs[0]
    elif coef_type == 'd':
        coeffs_mod[level] = coeffs[level]
    rec = pywt.waverec(coeffs_mod, wavelet)
    # Ajusta o tamanho para coincidir com a série original
    if len(rec) > orig_len:
        rec = rec[:orig_len]
    elif len(rec) < orig_len:
        rec = np.pad(rec, (0, orig_len - len(rec)), 'constant')
    return rec
component_names = []
# Componente de aproximação
approx_name = f'DWT_A{DECOMPOSITION_LEVEL}'
df_completo[approx_name] = reconstruct_component(coeffs, WAVELET_TYPE, DECOMPOSITION_LEVEL, 'a', len(ghi_series_continua))
component_names.append(approx_name)

# Componentes de detalhe
for level in range(1, DECOMPOSITION_LEVEL + 1):
    detail_name = f'DWT_D{level}'
    df_completo[detail_name] = reconstruct_component(coeffs, WAVELET_TYPE, level, 'd', len(ghi_series_continua))
    component_names.append(detail_name)

print(f"\nComponentes DWT gerados: {component_names}")
print("Cabeçalho do df_completo com os novos componentes DWT:")
print(df_completo[component_names].head())


# --- 3. ESTRUTURAÇÃO DA SÉRIE TEMPORAL (CRIAÇÃO DE FEATURES DEFASADAS) ---
def create_lagged_features(series, n_lags=10):
    """Cria N features defasadas para uma série temporal."""
    df_lags = pd.DataFrame(index=series.index)
    for i in range(1, n_lags + 1):
        df_lags[f'lag_{i}'] = series.shift(i)
    return df_lags

list_of_feature_dfs = []
for component_name in component_names:
    print(f"Criando features defasadas para o componente: {component_name}")
    component_series = df_completo[component_name]
    df_lags = create_lagged_features(component_series, n_lags=N_LAGS)
    # Renomear colunas para evitar conflitos (ex: A4_lag_1, D1_lag_1, etc.)
    df_lags = df_lags.add_prefix(f'{component_name}_')
    list_of_feature_dfs.append(df_lags)

# Combinar todas as features defasadas dos componentes DWT em um único DataFrame
df_dwt_features = pd.concat(list_of_feature_dfs, axis=1)

print("\nDimensões do DataFrame de features DWT:", df_dwt_features.shape)


# --- 4. COMBINAÇÃO FINAL E FILTRAGEM ---
# Adicionar outras features (ex: temperatura) ao DataFrame de features
df_dwt_features['temperature'] = df_completo['AIR TEMPERATURE - DRY BULB, HOURLY (°C)']

# Usar o índice do DataFrame JÁ FILTRADO POR ZÊNITE para selecionar os vetores de features relevantes
# Isso garante que estamos usando apenas os dados dos horários que nos interessam
df_final_features = df_dwt_features.loc[df_filtrado.index]

# Remover qualquer linha que ainda tenha NaNs (geralmente no início da série devido ao shift)
df_final_features.dropna(inplace=True)

print("\nDimensões do DataFrame final de features para clustering (após filtragem de zênite e remoção de NaNs):")
print(df_final_features.shape)
print("Cabeçalho do DataFrame final:")
print(df_final_features.head())


# --- 5. SALVAR O DATAFRAME DE FEATURES ---
output_path_features = 'data/features_para_clustering.parquet'
df_final_features.to_parquet(output_path_features)
print(f"\nDataFrame de features para clustering salvo em: {output_path_features}")


# --- 6. PLOT DE VERIFICAÇÃO (OPCIONAL) ---
# Plotar um componente DWT e a série original para um período específico
start_date = '2022-01-01'
end_date = '2022-02-07'

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
sample_data = df_completo.loc[start_date:end_date]

axes[0].plot(sample_data.index, ghi_series_continua.loc[start_date:end_date], label='GHI Medido Original')
axes[0].set_title('Série Original de GHI')
axes[0].legend()
axes[0].grid(True, linestyle=':')

axes[1].plot(sample_data.index, sample_data[f'DWT_A{DECOMPOSITION_LEVEL}'], label=f'A{DECOMPOSITION_LEVEL} (Tendência)', color='red')
axes[1].set_title(f'Componente de Aproximação (Tendência)')
axes[1].legend()
axes[1].grid(True, linestyle=':')

axes[2].plot(sample_data.index, sample_data['DWT_D1'], label='D1 (Variações Rápidas)', color='green')
axes[2].set_title('Componente de Detalhe de Alta Frequência')
axes[2].legend()
axes[2].grid(True, linestyle=':')

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import pywt
import os

def carregar_dados(caminho_completo, caminho_filtrado):
    """
    Carrega o DataFrame completo e o DataFrame filtrado por zênite.

    Args:
        caminho_completo (str): Caminho para o arquivo parquet com dados contínuos.
        caminho_filtrado (str): Caminho para o arquivo parquet com dados já filtrados.

    Returns:
        tuple: Uma tupla contendo (df_completo, df_filtrado).
    """
    print("--- 1. Carregando Dados ---")
    if not os.path.exists(caminho_completo):
        raise FileNotFoundError(f"Arquivo necessário não encontrado: {caminho_completo}")
    if not os.path.exists(caminho_filtrado):
        raise FileNotFoundError(f"Arquivo necessário não encontrado: {caminho_filtrado}")

    df_completo = pd.read_parquet(caminho_completo)
    df_filtrado = pd.read_parquet(caminho_filtrado)

    print(f"Dados completos carregados: {len(df_completo)} registros.")
    print(f"Dados filtrados por zênite carregados: {len(df_filtrado)} registros.")
    return df_completo, df_filtrado

def preparar_serie_para_dwt(df_completo):
    """
    Prepara a série GHI contínua para a aplicação da DWT, limpando e tratando os dados.

    Args:
        df_completo (pd.DataFrame): O DataFrame com os dados contínuos.

    Returns:
        pd.Series: A série GHI limpa e contínua.
    """
    print("\n--- 2. Preparando Série Temporal Contínua para DWT ---")
    ghi_series = df_completo['GLOBAL RADIATION (Kj/m²)'] / 3.6
    ghi_series.loc[ghi_series < 0] = 0
    ghi_series.interpolate(method='linear', inplace=True)
    ghi_series = ghi_series.bfill().ffill()
    print("Série GHI limpa e pronta para decomposição.")
    return ghi_series

def aplicar_dwt_e_reconstruir(serie, wavelet, nivel_decomp):
    """
    Aplica a DWT na série e reconstrói cada componente separadamente.

    Args:
        serie (pd.Series): A série temporal a ser decomposta.
        wavelet (str): O tipo de wavelet a ser usado (ex: 'db4').
        nivel_decomp (int): O nível de decomposição.

    Returns:
        pd.DataFrame: Um DataFrame onde cada coluna é um componente DWT reconstruído.
    """
    print(f"\n--- 3. Aplicando DWT (Wavelet: {wavelet}, Nível: {nivel_decomp}) ---")
    # Decompor a série
    coeffs = pywt.wavedec(serie, wavelet, level=nivel_decomp)
    orig_len = len(serie)
    
    df_componentes = pd.DataFrame(index=serie.index)
    component_names = []

    # Reconstruir componente de aproximação
    approx_name = f'DWT_A{nivel_decomp}'
    coeffs_mod = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    rec = pywt.waverec(coeffs_mod, wavelet)[:orig_len]
    df_componentes[approx_name] = rec
    component_names.append(approx_name)

    # Reconstruir componentes de detalhe
    for i in range(1, nivel_decomp + 1):
        detail_name = f'DWT_D{i}'
        # A lista de coeficientes é [cA, cD_n, cD_n-1, ..., cD_1]
        # Então o detalhe D_i está no índice (nivel_decomp - i + 1)
        level_index = nivel_decomp - i + 1
        coeffs_mod = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
        coeffs_mod[level_index] = coeffs[level_index]
        rec = pywt.waverec(coeffs_mod, wavelet)[:orig_len]
        df_componentes[detail_name] = rec
        component_names.append(detail_name)

    print(f"Componentes DWT gerados: {component_names}")
    return df_componentes

def criar_features_defasadas(df_componentes, n_lags):
    """
    Cria features defasadas (lags) para cada coluna do DataFrame de componentes.

    Args:
        df_componentes (pd.DataFrame): DataFrame com os componentes DWT.
        n_lags (int): O número de lags a serem criados.

    Returns:
        pd.DataFrame: Um DataFrame contendo todas as features defasadas.
    """
    print(f"\n--- 4. Criando {n_lags} Features Defasadas para cada Componente ---")
    list_of_feature_dfs = []
    for component_name in df_componentes.columns:
        series = df_componentes[component_name]
        df_lags = pd.DataFrame(index=series.index)
        for i in range(1, n_lags + 1):
            df_lags[f'{component_name}_lag_{i}'] = series.shift(i)
        list_of_feature_dfs.append(df_lags)
    
    df_dwt_features = pd.concat(list_of_feature_dfs, axis=1)
    print(f"Dimensões do DataFrame de features DWT defasadas: {df_dwt_features.shape}")
    return df_dwt_features

def combinar_e_filtrar_features(df_dwt_features, df_completo, df_filtrado):
    """
    Combina as features DWT com outras variáveis, filtra pelos horários de interesse
    e remove valores nulos.

    Args:
        df_dwt_features (pd.DataFrame): DataFrame com features DWT defasadas.
        df_completo (pd.DataFrame): DataFrame original para obter outras features (ex: temp).
        df_filtrado (pd.DataFrame): DataFrame filtrado para obter o índice de interesse.

    Returns:
        pd.DataFrame: O DataFrame final de features, pronto para o modelo.
    """
    print("\n--- 5. Combinando, Filtrando e Finalizando o Conjunto de Features ---")
    
    # Adicionar outras features
    df_dwt = df_dwt_features.copy()
    df_final = df_dwt_features.join(df_completo)
    
    # Usar o índice do DataFrame JÁ FILTRADO POR ZÊNITE para selecionar os vetores de features
    df_final = df_final.loc[df_filtrado.index]
    df_dwt = df_dwt.loc[df_filtrado.index]

    # Remover qualquer linha que ainda tenha NaNs (geralmente no início da série)
    df_final.dropna(inplace=True)
    df_dwt.dropna(inplace=True)
    
    print("Dimensões do DataFrame final de features (pós-filtro e limpeza):", df_final.shape)
    return df_final, df_dwt


def main():
    """
    Função principal para orquestrar o pipeline de engenharia de features.
    """
    print("Iniciando o script de engenharia de features com DWT...")
    
    # --- CONFIGURAÇÃO ---
    WAVELET_TYPE = 'db4'
    DECOMPOSITION_LEVEL = 4
    N_LAGS = 10
    ZENITH_ANGLE = 85

    # --- CAMINHOS ---
    path_zenith_filtered = f'data/Salinopolis_kt_Calculado_FusoCorrigido_Zenith{int(ZENITH_ANGLE)}.parquet'
    path_fuso_corrigido = 'data/Salinopolis_GHI_ClearSky_Calculado_FusoCorrigido.parquet'
    output_path_features = 'data/features_para_clustering02.parquet'
    output_path_dwt = 'data/dwtData.parquet'

    # --- PIPELINE ---
    # 1. Carregar dados
    df_completo, df_filtrado = carregar_dados(path_fuso_corrigido, path_zenith_filtered)
    
    # 2. Preparar série para DWT
    ghi_series_continua = preparar_serie_para_dwt(df_completo)

    # 3. Aplicar DWT e reconstruir componentes
    df_componentes = aplicar_dwt_e_reconstruir(ghi_series_continua, WAVELET_TYPE, DECOMPOSITION_LEVEL)
    
    # 4. Criar features defasadas (janela deslizante)
    df_dwt_features = criar_features_defasadas(df_componentes, N_LAGS)
    
    # 5. Combinar, filtrar e finalizar
    df_final_features, df_dwt_final = combinar_e_filtrar_features(df_dwt_features, df_completo, df_filtrado)

    # --- SALVAR RESULTADO ---
    print(f"\n--- 6. Salvando Resultado ---")
    df_final_features.to_parquet(output_path_features)
    df_dwt_final.to_parquet(output_path_dwt)
    print(f"DataFrame de features pronto para clustering salvo em: {output_path_features}")
    print("\nCabeçalho do DataFrame final:")
    print(df_final_features.head())
    print("\nScript finalizado com sucesso!")


if __name__ == "__main__":
    main()
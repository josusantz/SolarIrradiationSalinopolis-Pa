import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import warnings

# Ignorar avisos futuros sobre n_init que pode mudar o padrão no futuro
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster._kmeans')

# --- 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
features_path = 'data/features_para_clustering.parquet'

if not os.path.exists(features_path):
    raise FileNotFoundError(f"Arquivo de features não encontrado: '{features_path}'. Certifique-se de que a etapa anterior foi executada.")

df_features = pd.read_parquet(features_path)
print(f"DataFrame de features carregado com sucesso. Dimensões: {df_features.shape}")

# Escalonamento das features (essencial para K-Means)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_features)
print("Features escalonadas com StandardScaler.")


# --- 2. CÁLCULO DAS MÉTRICAS DE CLUSTERING ---
# Definir o intervalo de k a ser testado. Um intervalo menor (ex: até 30) é geralmente suficiente e mais rápido.
possible_k_values = range(2, 31) 

# Listas para armazenar os resultados das métricas
inertia_values = []
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []

print("\nIniciando o cálculo das métricas para diferentes valores de k...")
for k in possible_k_values:
    # Inicializar e treinar o modelo K-Means
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(features_scaled)
    
    # Obter os rótulos dos clusters para cada ponto de dado
    cluster_labels = kmeans.labels_
    
    # 1. Inércia (Soma dos quadrados das distâncias intra-cluster)
    inertia = kmeans.inertia_
    inertia_values.append(inertia)
    
    # 2. Silhouette Score (Pico é melhor)
    # Requer pelo menos 2 clusters.
    silhouette = silhouette_score(features_scaled, cluster_labels)
    silhouette_scores.append(silhouette)
    
    # 3. Calinski-Harabasz Score (Pico é melhor)
    calinski = calinski_harabasz_score(features_scaled, cluster_labels)
    calinski_harabasz_scores.append(calinski)
    
    # 4. Davies-Bouldin Score (Vale é melhor)
    davies = davies_bouldin_score(features_scaled, cluster_labels)
    davies_bouldin_scores.append(davies)
    
    print(f"k={k:2d} | Inércia: {inertia:10.2f} | Silhouette: {silhouette:.4f} | Calinski-Harabasz: {calinski:8.2f} | Davies-Bouldin: {davies:.4f}")

# Criar um DataFrame com os resultados para fácil visualização
results_df = pd.DataFrame({
    'k': list(possible_k_values),
    'inertia': inertia_values,
    'silhouette': silhouette_scores,
    'calinski_harabasz': calinski_harabasz_scores,
    'davies_bouldin': davies_bouldin_scores
})
results_df.to_csv("clusterOtimalValidation.csv")

print("\nCálculo concluído.")
print("Resultados:")
print(results_df)


# --- 3. VISUALIZAÇÃO DAS MÉTRICAS ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análise do Número Ótimo de Clusters (k)', fontsize=16)

# Plot 1: Método do Cotovelo (Inércia)
axes[0, 0].plot(results_df['k'], results_df['inertia'], marker='o', linestyle='-')
axes[0, 0].set_title('Método do Cotovelo (Inércia)')
axes[0, 0].set_xlabel('Número de Clusters (k)')
axes[0, 0].set_ylabel('Inércia (WCSS)')
axes[0, 0].grid(True, linestyle=':')
axes[0, 0].text(0.95, 0.95, 'Procure o "cotovelo"', transform=axes[0, 0].transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Plot 2: Silhouette Score
axes[0, 1].plot(results_df['k'], results_df['silhouette'], marker='o', linestyle='-', color='g')
axes[0, 1].set_title('Silhouette Score')
axes[0, 1].set_xlabel('Número de Clusters (k)')
axes[0, 1].set_ylabel('Score')
axes[0, 1].grid(True, linestyle=':')
axes[0, 1].text(0.95, 0.05, 'Procure o PICO', transform=axes[0, 1].transAxes, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5))

# Plot 3: Calinski-Harabasz Score
axes[1, 0].plot(results_df['k'], results_df['calinski_harabasz'], marker='o', linestyle='-', color='r')
axes[1, 0].set_title('Calinski-Harabasz Score')
axes[1, 0].set_xlabel('Número de Clusters (k)')
axes[1, 0].set_ylabel('Score')
axes[1, 0].grid(True, linestyle=':')
axes[1, 0].text(0.95, 0.05, 'Procure o PICO', transform=axes[1, 0].transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.5))

# Plot 4: Davies-Bouldin Score
axes[1, 1].plot(results_df['k'], results_df['davies_bouldin'], marker='o', linestyle='-', color='purple')
axes[1, 1].set_title('Davies-Bouldin Score')
axes[1, 1].set_xlabel('Número de Clusters (k)')
axes[1, 1].set_ylabel('Score')
axes[1, 1].grid(True, linestyle=':')
axes[1, 1].text(0.95, 0.95, 'Procure o VALE', transform=axes[1, 1].transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='thistle', alpha=0.5))
plt.savefig('COV.jpeg')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carregar o DataFrame de features que você criou
path_features = 'data/features_para_clustering02.parquet'
path_dwt = 'data/dwtData.parquet'
df_features = pd.read_parquet(path_features)


selectedFeatures = ['TOTAL PRECIPITATION, HOURLY (mm)',
       'ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)',
       'ATMOSPHERIC PRESSURE REDUCED TO SEA LEVEL, AUT (mB)',
       'ATMOSPHERIC PRESSURE MAX. IN THE PREVIOUS HOUR (AUT) (mB)',
       'ATMOSPHERIC PRESSURE MIN. IN THE PREVIOUS HOUR (AUT) (mB)',
       'GLOBAL RADIATION (Kj/m²)', 'STATION CPU TEMPERATURE (°C)',
       'AIR TEMPERATURE - DRY BULB, HOURLY (°C)', 'DEW POINT TEMPERATURE (°C)',
       'MAXIMUM TEMPERATURE IN THE PREVIOUS HOUR (AUT) (°C)',
       'MINIMUM TEMPERATURE IN THE PREVIOUS HOUR (AUT) (°C)',
       'MAXIMUM DEW POINT TEMPERATURE IN THE PREVIOUS HOUR (AUT) (°C)',
       'MINIMUM DEW POINT TEMPERATURE IN THE PREVIOUS HOUR (AUT) (°C)',
       'STATION BATTERY VOLTAGE (V)',
       'RELATIVE HUMIDITY MAX. IN THE PREVIOUS HOUR (AUT) (%)',
       'RELATIVE HUMIDITY MIN. IN THE PREVIOUS HOUR (AUT) (%)',
       'AIR RELATIVE HUMIDITY, HOURLY (%)',
       'WIND, HOURLY DIRECTION (gr) (° (gr))', 'WIND, MAXIMUM GUST (m/s)',
       'WIND, HOURLY SPEED (m/s)', 'precipitable_water', 'linke_turbidity',
       'pressure_pascals', 'apparent_zenith', 'solar_azimuth', 'dni_extra',
       'airmass_absolute', 'GHIcs_Ineichen', 'dni', 'dhi',
       'GHI_measured_Wm2_plot']

df_features = df_features[selectedFeatures]

print("DataFrame de features carregado. Dimensões:", df_features.shape)
print("Cabeçalho dos dados:")
print(df_features.head())

# 1. Escalonamento dos Dados
# É crucial para que o algoritmo não seja enviesado por features com escalas diferentes.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_features)
# 2. Encontrar o número ideal de clusters com o Método do Cotovelo
print("\nIniciando o Método do Cotovelo para encontrar o 'k' ideal...")

inertia_values = []
possible_k = range(2, 101) # Testaremos de 2 a 15 clusters

for k in possible_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(features_scaled)
    inertia_values.append(kmeans.inertia_)
    print(f"Inércia para k={k}: {kmeans.inertia_:.2f}")

# 3. Plotar o gráfico do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(possible_k, inertia_values, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (Within-cluster sum of squares)')
plt.title('Método do Cotovelo para Encontrar o k Ideal')
plt.xticks(possible_k)
plt.grid(True)
plt.savefig('knnDwt101.png')
plt.show()
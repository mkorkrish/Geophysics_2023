import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # This line needs to be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_excel("Data Test MK Up and Lr BKK.xlsx")

# Convert '<0.01' to 0.005 and non-numeric values to NaN
df = df.replace('<0.01', 0.005)
df = df.apply(pd.to_numeric, errors='coerce')

# Selecting metal concentration columns
metals = df.loc[:, 'Na %':'Mo (ppm)']

# Descriptive statistics
desc_stats = metals.describe()

# Correlation analysis
corr_matrix = metals.corr()

# Print descriptive statistics and correlation matrix
print(desc_stats)
print(corr_matrix)

# Heatmap of correlations
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.savefig('heatmap.png')
plt.show()

# Histograms for each metal
metals.hist(figsize=(20, 20), bins=50)
plt.tight_layout()
plt.savefig('histograms.png')
plt.show()

# Box plots for each metal
metals.boxplot(figsize=(20, 10), rot=90)
plt.tight_layout()
plt.savefig('boxplots.png')
plt.show()

# Pairwise scatter plots
sns.pairplot(metals.sample(1000))  # Sample 1000 points for efficiency
plt.savefig('pairplot.png')
plt.show()

# Time series plot for Aluminum
plt.figure(figsize=(15, 5))
plt.plot(df['Al %'])
plt.title('Time Series of Aluminum Concentration')
plt.xlabel('Index (Assumed to be Time)')
plt.ylabel('Al %')
plt.savefig('timeseries.png')
plt.show()

# Spatial plot for Aluminum
plt.figure(figsize=(10, 10))
plt.scatter(df['X_Long'], df['Y_Lat'], c=df['Al %'], cmap='viridis')
plt.title('Spatial Distribution of Aluminum Concentration')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Al %')
plt.savefig('spatial.png')
plt.show()

# Standardize the metals data, drop rows with missing values
metals_no_na = metals.dropna()
scaler = StandardScaler()
metals_scaled = scaler.fit_transform(metals_no_na)

# Perform PCA
pca = PCA()
principalComponents = pca.fit_transform(metals_scaled)

# Scree plot
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig('screeplot.png')
plt.show()

# Convert to a DataFrame
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC' + str(i) for i in range(1, principalComponents.shape[1] + 1)])

# Visualize the first two principal components
plt.figure(figsize=(10, 10))
plt.scatter(principalDf['PC1'], principalDf['PC2'])
plt.title('2 Component PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('pca.png')
plt.show()

# Perform K-means clustering
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(metals_scaled)

# Plot the clusters
plt.scatter(principalDf['PC1'], principalDf['PC2'], c=clusters, cmap='viridis')
plt.title('K-means Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('clusters.png')
plt.show()

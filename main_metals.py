import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Read the Excel spreadsheet into a pandas DataFrame
df = pd.read_excel("geophysics_data.xlsx")

# Replace non-numeric values with NaN
df.replace('<0.01', np.nan, inplace=True)  # Replace '<0.01' with NaN, adjust as needed

# Descriptive Statistics
numeric_vars = ["Na %", "Mg %", "Al %", "Si %", "P %", "S %", "K % ", "%Ca", "%Ti", "%Mn", "%Fe"]  # Add other numeric variables as needed

# Calculate and display correlation matrix
corr_matrix = df[numeric_vars].corr()
print("Correlation Matrix:")
print(corr_matrix)
print()

# Plotting correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("Correlation Matrix", fontsize=14)
plt.xticks(np.arange(len(numeric_vars)), numeric_vars, rotation=45)
plt.yticks(np.arange(len(numeric_vars)), numeric_vars)
plt.show()

for var in numeric_vars:
    desc_stats = df[var].describe()
    print(f"Descriptive Statistics for {var}:")
    print(desc_stats)
    print()
    
    # Plotting histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df.dropna(subset=[var])[var], bins=10, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel(f"{var} (%)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution of {var}", fontsize=14)
    plt.grid(True)
    plt.show()

    # Hypothesis Testing (t-test)
    grouped_data = df.groupby("Formation")

    upper_bakken = grouped_data.get_group("Upper Bakken").copy()
    lower_bakken = grouped_data.get_group("Lower Bakken").copy()

    # Convert columns to numeric data type
    upper_bakken.loc[:, var] = pd.to_numeric(upper_bakken[var], errors="coerce")
    lower_bakken.loc[:, var] = pd.to_numeric(lower_bakken[var], errors="coerce")

    t_stat, p_value = stats.ttest_ind(upper_bakken[var].dropna(), lower_bakken[var].dropna(), nan_policy='omit')
    print(f"Hypothesis Testing for {var}:")
    print("t-statistic:", t_stat)
    print("p-value:", p_value)
    print()

    # Regression Analysis (Linear Regression)
    X = df.dropna(subset=[var])[["Na %", "Mg %", "Al %", "Si %", "P %", "S %", "K % ", "%Ca", "%Ti", "%Mn","%Fe"]]  # Independent variables
    y = df.dropna(subset=[var])[var]  # Dependent variable

    imputer = SimpleImputer(strategy="mean")  # or "median", "most_frequent", etc.
    X_imputed = imputer.fit_transform(X)

    model = LinearRegression()
    model.fit(X_imputed, y)

    coefficients = pd.DataFrame({"Variable": X.columns, "Coefficient": model.coef_})
    print(f"Linear Regression for {var}:")
    print(coefficients)
    print()

    # Plotting linear regression
    y_pred = model.predict(X_imputed)

    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Actual Value", fontsize=12)
    plt.ylabel("Predicted Value", fontsize=12)
    plt.title(f"Linear Regression for {var}", fontsize=14)
    plt.grid(True)
    plt.show()

# Cluster Analysis (K-means Clustering)
X = df[numeric_vars]

imputer = SimpleImputer(strategy="mean")  # or "median", "most_frequent", etc.
X_imputed = imputer.fit_transform(X)

kmeans = KMeans(n_clusters=3)  # Choose the desired number of clusters
kmeans.fit(X_imputed)

cluster_labels = kmeans.labels_
df["Cluster"] = cluster_labels
print("Cluster Analysis:")
print(df["Cluster"].value_counts())
print()

# Principal Component Analysis (PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

pca = PCA(n_components=2)  # Choose the desired number of principal components
pca_scores = pca.fit_transform(X_scaled)

df["PC1"] = pca_scores[:, 0]
df["PC2"] = pca_scores[:, 1]

# Data Visualization (PCA Plot)
plt.figure(figsize=(8, 6))
plt.scatter(df["PC1"], df["PC2"], c=df["Cluster"], cmap='viridis')
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.title("PCA Plot", fontsize=14)
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

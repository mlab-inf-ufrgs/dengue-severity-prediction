import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from utils.constants import NUMERIC_ATTRS

df = pd.read_csv("../data/3_gold/dataset-processed.csv")
df = df.sample(n=50000, random_state=42)

# Normalize numeric attributes
for col in NUMERIC_ATTRS:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

X = df.drop("severity", axis=1)
y = df["severity"]

pca = PCA(n_components=2) # Reduce to 2 principal components
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))

colors = {'low_risk': 'blue', 'alarm': 'orange', 'severe': 'red'}
markers = {'low_risk': 'o', 'alarm': '^', 'severe': 's'}

# Plot each class
for label_name in y.unique():
    # Filter data for the current label
    idx = y == label_name
    
    # Plot the 2D PCA projection
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        c=colors.get(label_name, 'gray'),
        marker=markers.get(label_name, '*'),
        label=f'{label_name} ({Counter(y)[label_name]} samples)',
        alpha=0.7,
        s=50
    )

plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('2D PCA Projection of Dataset by Severity Label')
plt.legend(title='Class Distribution')
plt.grid(True)
plt.show()

print(f"Explained variance ratio by principal components: {pca.explained_variance_ratio_}")
print(f"Total explained variance by 2 components: {pca.explained_variance_ratio_.sum()*100:.2f}%")
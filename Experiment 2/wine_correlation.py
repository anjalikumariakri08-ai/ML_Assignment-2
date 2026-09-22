import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Correlation matrix
corr = df.corr()

print("Correlation Matrix:")
print(corr)

# Find strongest positive correlation
corr_values = corr.where(
    ~np.eye(corr.shape[0], dtype=bool)
)

max_corr = corr_values.max().max()

for col in corr.columns:
    for row in corr.index:
        if corr.loc[row, col] == max_corr:
            print("\nStrongest Positive Correlation:")
            print(row, "and", col)
            print("Correlation:", max_corr)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Wine Dataset Correlation Heatmap")
plt.tight_layout()
plt.savefig("wine_correlation_heatmap.png")
plt.show()
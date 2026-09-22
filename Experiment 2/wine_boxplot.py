import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Plot boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)

plt.title("Boxplots of Wine Dataset")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("wine_boxplots.png")
plt.show()
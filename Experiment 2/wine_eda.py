import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
print("First 5 Rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nCorrelation:")
print(df.corr(numeric_only=True))
plt.figure(figsize=(6, 4))
sns.histplot(df["alcohol"], kde=True)
plt.title("Alcohol Distribution")
plt.savefig("wine_histogram.png")
plt.show()
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="alcohol", y="malic_acid", hue="target")
plt.title("Alcohol vs Malic Acid")
plt.savefig("wine_scatter.png")
plt.show()

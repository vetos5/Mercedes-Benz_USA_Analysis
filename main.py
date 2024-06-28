import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import AgglomerativeClustering

path = 'E:/DATASETS/usa_mercedes_benz_prices.csv'
df = pd.read_csv(path)

print(df.dtypes)

df['Name'] = df['Name'].str.strip()
df['Name'] = df['Name'].str.title()
df['Name'] = df['Name'].astype('string')
df['Mileage'] = df['Mileage'].str.replace(',', '').str.replace(' mi.', '').astype(float)
df['Price'] = df['Price'].str.replace(',', '').str.replace('$', '')
df['Price'] = df['Price'].replace('Not Priced', np.nan).astype(float)
df = df.dropna(subset=['Review Count'])
df['Review Count'] = df['Review Count'].str.replace(',', '').astype(int)

print(df.dtypes)

print(df.describe())

plt.hist(df['Price'], bins=50)
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()

sns.scatterplot(x='Mileage', y='Price', hue='Rating', data=df, palette='viridis')
plt.xlabel('Mileage (miles)')
plt.ylabel('Price ($)')
plt.title('Mileage vs Price')
plt.legend(title='Rating')
plt.show()

numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

sns.set()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


df.dropna(inplace=True)

features = ['Mileage', 'Price', 'Rating']

agg_cluster = AgglomerativeClustering(n_clusters=3)
cluster_labels = agg_cluster.fit_predict(df[features])
plt.scatter(df['Mileage'], df['Price'], c=cluster_labels, cmap='viridis')
legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Luxury', markerfacecolor=plt.cm.viridis(0), markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Mid-priced', markerfacecolor=plt.cm.viridis(0.5),
                   markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Used', markerfacecolor=plt.cm.viridis(1.0), markersize=10)]

plt.legend(handles=legend_handles, loc='upper right')
plt.xlabel('Mileage (miles)')
plt.ylabel('Price ($)')
plt.title('Car Model Clusters')
plt.show()




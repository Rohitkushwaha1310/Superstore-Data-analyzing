import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("superstore.csv")

print(df.head())
sns.set_style('whitegrid')
sns.set_palette('husl')
print("ready")

plt.figure(figsize=(10,6))
sns.boxplot(x='Category', y='Sales', data=df)
plt.title('Sales Distrcation by category', fontsize=14)
plt.xlabel('Category')
plt.ylabel('Sales')
# plt.show()

plt.figure(figsize=(8,6))

corr =df[['Sales', 'Profit', 'Quantity', 'Discount']].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap= 'coolwarm',
    linewidths=0.5

)
plt.title('Correlation Heatmap', fontsize=14)
# plt.show()

#pairplot
sns.pairplot(
    df[['Sales', 'Profit', 'Quantity', 'Discount']],
    diag_kind='kde',
    plot_kws={'alpha':0.3}

)

plt.suptitle('Pairplot-All relations', y=1.02, fontsize=14)
# plt.show()

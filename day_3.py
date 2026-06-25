import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("data.csv",encoding="latin1")
# print(df.shape)
# print(df.dtypes)
# print("satatical summary")
# print(df.describe)
# print("mean sales", round(df['Sales'].mean(),2))
# print("max profit", round(df['Profit'].max(),2))


# print("sales stats:")
# print(df['Sales'].describe())
# print("\n salesSkewness", round(df['Sales'].skew(),2))
# print("\n profitSkewness", round(df['Profit'].skew(),2))

# print("order by category")
# print(df['Category'].value_counts())

corr = df[['Sales', 'Profit', 'Quantity', 'Discount']].corr()
# print(corr)

# print("correleationn with profit")
# print(corr['Profit'].sort_values(ascending=False))

Q1= df['Sales'].quantile(0.25)
Q3= df['Sales'].quantile(0.75)
IQR = Q3-Q1
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

outliers = df[(df['Sales']<lower) | (df['Sales']>upper)]
print(f"Q1:{Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"lower bound: {lower}")
print(f"upper bound: {upper}")
print(f"NUmbr od outliers: {len(outliers)}")



#visualisatio
fig, axes = plt.subplots(2,2, figsize= (14,10))
fig.suptitle('Superstore EDZ Dashboard', fontsize=16)

axes[0,0].hist(df["Sales"], bins=50, color='steelblue', edgecolor='white')
axes[0,0].set_title('Sales Distribution')
axes[0,0].set_xlabel('Sales')

cat_profit = df.groupby('Category')['Profit'].sum()
axes[0,1].bar(cat_profit.index, cat_profit.values, color=['steelblue', 'orange', 'green'])
axes[0,1].set_title('Total profit by category')

axes[1,0].scatter(df['Discount'] ,df['Profit'], alpha=0.3, color='tomato')
axes[1,0].set_title('Discount vs Profit')
axes[1,0].set_xlabel('Discount')
axes[1,0].set_ylabel('Profit')


region_sales= df.groupby('Region')['Sales'].sum()
axes[1,1].pie(region_sales, labels= region_sales.index, autopct='%1.1f%%')
axes[1,1].set_title('Sales By region')
plt.tight_layout
plt.show()



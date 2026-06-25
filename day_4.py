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



#sales trend over time
df['Order Date']= pd.to_datetime(df['Order Date'])

df['Year']= df['Order Date'].dt.year
df['Month']= df['Order Date'].dt.month


monthly_sales= df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
monthly_sales['Date']= pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))

plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Sales', data=monthly_sales, color='steelblue', linewidth=2)
plt.title('Monthly sales trend', fontsize= 14)
plt.xlabel('Date')
plt.ylabel('Total sales')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

#year wise sales
yearly_sales= df.groupby('Year')['Sales'].agg(['sum','mean', 'count']).reset_index()
yearly_sales.columns=['Year','Total Sales','Avg Sales','Orders']
print(yearly_sales)

plt.figure(figsize=(10,5))
sns.barplot(x='Year', y='Total Sales', data=yearly_sales, palette='husl')
plt.title('Total sales by year', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Total sales')

for i, row in yearly_sales.iterrows():
    plt.text(i, row['Total Sales']+1000,
    f"${row['Total Sales']:,.0f}",
    ha='center', fontsize=10)


plt.tight_layout()
# plt.show()

fig, axes= plt.subplots(1,3, figsize=(18,5))

# rofit by category
cat_profit= df.groupby('Category')['Profit'].sum()

axes[0].bar(cat_profit.index, cat_profit.values,color=['green' if x>0 else 'red' for x in cat_profit.values])
axes[0].set_title('which category makes money')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Total profit')

#Discount imapct
axes[1].scatter(df['Discount'], df['Profit'], alpha=0.2, color='tomato')
axes[1].set_title('Does Discount hurt profit')
axes[1].set_xlabel('Discount')
axes[1].set_ylabel('Profit')

# year wise growth
yearly = df.groupby('Year')['Sales'].sum()
axes[2].plot(yearly.index, yearly.values, marker ='o', color='steelblue', linewidth=2)
axes[2].fill_between(yearly.index, yearly.values, alpha=0.2)
axes[2].set_title('Is bussiness growing')
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Total Sales')

plt.tight_layout()
plt.show()







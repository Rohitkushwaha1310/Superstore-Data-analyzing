import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("superstore.csv")
df= df.dropna(subset=['Sales', 'Order Date'])



df['Order Date']= pd.to_datetime(df['Order Date'])


df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter
df['DayOfWeek'] = df['Order Date'].dt.dayofweek
df['DayOfMonth'] = df['Order Date'].dt.day
df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
df['DaysToShip']= (pd.to_datetime(df['Ship Date']) - pd.to_datetime(df['Order Date'])).dt.days


# print(" new date features")
# print(df[['Order Date', 'Year', 'Month', 
#           'Quarter', 'DayOfWeek', 
#           'IsWeekend', 'DaysToShip']].head(10))



# analyse new features

fig, axes = plt.subplots(2,2, figsize= (14, 10))     
fig.suptitle('Date Feature Analysis', fontsize=16)

# sales by month
monthly = df.groupby('Month')['Sales'].mean()
axes[0,0].bar(monthly.index, monthly.values, color ='steelblue')
axes[0,0].set_title('Avg sales by month')
axes[0,0].set_xlabel('Month')

# sales by quater
quarterly = df.groupby('Quarter')['Sales'].sum()
axes[0,1].bar(quarterly.index, quarterly.values, color='green')
axes[0,1].set_title('Toatal sales by quarter')


# sales weekend s weekdays
weekend = df.groupby('IsWeekend')['Sales'].sum()
axes[1,0].bar(['Weekday', 'Weekend'], weekend.values,
                color = ['Steelblue', 'orange'])
axes[1,0].set_title('AVG SALales: weekend vs weekday')


axes[1,1].hist(df['DaysToShip'].dropna(), bins=20,
               color='purple', edgecolor='white')
axes[1,1].set_title('Days to Ship Distribution')

# plt.tight_layout()
# plt.show()

# print("\n=== KEY STATS ===")
# print(f"Best sales month  : {monthly.idxmax()}")
# print(f"Best quarter      : Q{quarterly.idxmax()}")
# print(f"Avg days to ship  : {df['DaysToShip'].mean():.1f}")
# print(f"Weekend orders    : {df['IsWeekend'].sum()}")    

df['Sales_Log'] = np.log1p(df['Sales'])
df['Profit_Log']= np.log1p(df['Profit'].clip(lower=0)) 
df['Sales_per_Qty']= df['Sales']/ df['Quantity']
df['Discount_Impact']= df['Sales']* df['Discount']
df['Profit_per_day']= df['Profit']/ (df['DaysToShip']+1)
df['Sales_Category']= pd.cut(df['Sales'],
    bins = [0, 100, 500, 1000, 10000],
    labels= ['Low', 'Medium', 'High', 'Premium'])


fig , axes =plt.subplots(1,3, figsize=(15,5))
axes[0].hist(df['Sales'], bins=50, color = 'steelblue', edgecolor='white')
axes[0].set_title('Orginal Sales')
axes[1].hist(df['Sales_Log'], bins=50, color='green', edgecolor='white')
axes[1].set_title('Log Sales')
df['Sales_Category'].value_counts().plot(kind='bar', ax=axes[2], color='orange')
axes[2].set_title('Log Sales')
plt.tight_layout()
plt.show()


print(f"Original Skewness : {df['Sales'].skew():.2f}")
print(f"Log Skewness      : {df['Sales_Log'].skew():.2f}")
print(df['Sales_Category'].value_counts())
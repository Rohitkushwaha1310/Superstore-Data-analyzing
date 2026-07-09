import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
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
# plt.show()


# print(f"Original Skewness : {df['Sales'].skew():.2f}")
# print(f"Log Skewness      : {df['Sales_Log'].skew():.2f}")
# print(df['Sales_Category'].value_counts())


#aggreation Feature
customer_agg = df.groupby('Customer Name').agg(
    Total_Sales = ('Sales',   'sum'),
    Total_orders= ('Order ID', 'count'),
    Avg_Order_Value = ('Sales', 'mean'),
    Total_Profit = ('Profit', 'sum'),
    Avg_Discount = ('Discount', 'mean')
).reset_index()

customer_agg['Profit_Rate']=(
    customer_agg['Total_Profit']/customer_agg['Total_Sales']*100
).round(2)

# print(" customer aggregations")
# print(customer_agg.sort_values(
#     'Total_Sales', ascending=False
# ).head(10))


#product leve aggregations
product_agg = df.groupby('Sub-Category').agg(
    Total_Sales =('Sales',  'sum'),
    Avg_Profit  =('Profit', 'mean'),
    Order_Count =('Order ID','count'),
    Avg_Discount=('Discount','mean')
).reset_index().sort_values('Total_Sales', ascending=False)

print("\n=== PRODUCT AGGREGATIONS ===")
print(product_agg.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh(product_agg['Sub-Category'],
             product_agg['Total_Sales'],
             color='steelblue')
axes[0].set_title('Total Sales by Sub-Category')

axes[1].barh(product_agg['Sub-Category'],
             product_agg['Avg_Profit'],
             color=['green' if x > 0 else 'red' 
                    for x in product_agg['Avg_Profit']])
axes[1].set_title('Avg Profit by Sub-Category')
# plt.tight_layout()
# plt.show()

## buildinng ml model by feature engineerinng 
df_ml = pd.get_dummies(df[[
    'Sales', 'Quantity','Discount',
    'Month', 'Quarter', 'DaysToShip',
    'IsWeekend', 'Sales_per_Qty',
    'Discount_Impact', 'Category',
    'Region', 'Segment'
]].dropna(), columns=['Category', 'Region', 'Segment'], drop_first=True)

x= df_ml.drop('Sales', axis=1)
y= df_ml['Sales']
X_train , X_test, y_train, y_test= train_test_split(
     x,y, test_size=0.2, random_state=42
)

rf_basic = RandomForestRegressor(n_estimators=100, random_state=42)
rf_basic.fit(X_train[['Quantity', 'Discount']],y_train)
basic_r2 = r2_score(y_test, rf_basic.predict(X_test[['Quantity', 'Discount']]))


# mmodel with enginner
rf_eng = RandomForestRegressor(n_estimators=100, random_state=42)
rf_eng.fit(X_train, y_train)
eng_r2 = r2_score(y_test, rf_eng.predict(X_test))
eng_rmse = np.sqrt(mean_squared_error(
    y_test, rf_eng.predict(X_test)))

print("=" * 45)
print(f"Basic RF R²      : {basic_r2:.4f}")
print(f"Engineered RF R² : {eng_r2:.4f}")
print(f"Improvement      : {((eng_r2-basic_r2)/abs(basic_r2)*100):.1f}%")
print(f"RMSE             : {eng_rmse:.2f}")



importance = pd.DataFrame({
    'Feature'    : x.columns,
    'Importance' : rf_eng.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

print("\n=== TOP 10 FEATURES ===")
print(importance.to_string())

plt.figure(figsize=(10, 6))
plt.barh(importance['Feature'],
         importance['Importance'],
         color='steelblue')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()

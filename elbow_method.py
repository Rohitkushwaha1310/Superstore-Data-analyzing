import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("superstore.csv")
df= df.dropna(subset=['Sales','Profit','Discount','Quantity'])


customer= df.groupby('Customer Name').agg(
    Total_Sales   =('Sales',    'sum'),
    Total_Profit  =('Profit',   'sum'),
    Order_Count   =('Order ID', 'count'),
    Avg_Discount  =('Discount', 'mean'),
    Avg_Quantity  =('Quantity', 'mean')

).reset_index()

print("Cluster feature ")
print(customer.describe())

# scle feature

features = ['Total_Sales','Total_Profit',
            'Order_Count','Avg_Discount','Avg_Quantity']



x= customer[features]
scaler = StandardScaler()
x_scaled = scaler.fit_transform()

print(f"\n Original sales group :{x['Total_Sales'].min():.0f}- {x['Total_sales'].max():.0f}")
print(f"scales sales range: {x_scaled[:,0].min():.2f} - {X_scaled[:,0].max():.2f}")



#elbbow methos 
inertias = []
silhouettes= []
k_range = range(2,11)


for k in k_range:
    km= KMeans(n_clusters=k, random_state=42)
            
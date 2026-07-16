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

# print("Cluster feature ")
# print(customer.describe())

# scle feature

features = ['Total_Sales','Total_Profit',
            'Order_Count','Avg_Discount','Avg_Quantity']



x= customer[features]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print(f"\nOriginal Sales range : {x['Total_Sales'].min():.0f} - {x['Total_Sales'].max():.0f}")
print(f"scales sales range: {x_scaled[:,0].min():.2f} - {x_scaled[:,0].max():.2f}")



#elbbow methos 
inertias = []
silhouettes= []
k_range = range(2,11)


for k in k_range:
    km= KMeans(n_clusters=k, random_state=42)
    km.fit(x_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(x_scaled, km.labels_))
    print(f"k={k}: Inertia = {km.inertia_:.0f}, Silhouette= {silhouette_score(x_scaled, km.labels_):.3f}")



#visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method — Find Optimal K')
axes[0].grid(True)

axes[1].plot(k_range, silhouettes, 'ro-', linewidth=2)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs K')
axes[1].grid(True)

# plt.tight_layout()
# plt.show()



optimal_k= 5
km_final= KMeans(n_clusters=optimal_k,
                    random_state=42, n_init=10)

customer['Cluster']= km_final.labels_


print(" cluster distribution")
print(customer['Cluster'].value_counts().sort_index())


# cluster profiles
profile=customer.groupby('Cluster')[features].mean()
print("cluster profiles")
print(profile.round(2).to_string())

#visualize cluster

fig, axes= plt.subplots(2,2, figsize=(14,10))
fig.suptitle('Customer cluster analysis')


#sales vs profit by cluster

scatter = axes[0,0].scatter(
    customer['Total_Sales'],
    customer['Total_Profile'],
    c= customer['Cluster'],
    cmap='viridis', alpha=0.6
)

axes[0,0].set_xlabel('Total Sales')
axes[0,0].set_ylabel('Total Profit')
axes[0,0].set_title('Sales vs prfit by cluster')
plt.colorbar(scatter, ax=axes[0,0])

#order count vs 
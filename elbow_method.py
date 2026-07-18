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

# print(f"\nOriginal Sales range : {x['Total_Sales'].min():.0f} - {x['Total_Sales'].max():.0f}")
# print(f"scales sales range: {x_scaled[:,0].min():.2f} - {x_scaled[:,0].max():.2f}")



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

km_final.fit(x_scaled)                    

customer['Cluster'] = km_final.labels_


print(" cluster distribution")
print(customer['Cluster'].value_counts().sort_index())


# cluster profiles
profile=customer.groupby('Cluster')[features].mean()
# print("cluster profiles")
# print(profile.round(2).to_string())

#visualize cluster

fig, axes= plt.subplots(2,2, figsize=(14,10))
fig.suptitle('Customer cluster analysis')


#sales vs profit by cluster

scatter = axes[0,0].scatter(
    customer['Total_Sales'],
    customer['Total_Profit'],
    c= customer['Cluster'],
    cmap='viridis', alpha=0.6
)

axes[0,0].set_xlabel('Total Sales')
axes[0,0].set_ylabel('Total Profit')
axes[0,0].set_title('Sales vs prfit by cluster')
plt.colorbar(scatter, ax=axes[0,0])

#order count vs  avg discounnt

axes[0,1].scatter(
    customer['Order_Count'],
    customer['Avg_Discount'],
    c=customer['Cluster'],
    cmap='viridis', alpha=0.6
)
axes[0,1].set_xlabel('Order count')
axes[0,1].set_ylabel('Avg discounnt')
axes[0,1].set_title('Orders vs Discount by cluster')


#cluster Sizes
cluster_sizes = customer['Cluster'].value_counts().sort_index()
axes[1,0].bar(cluster_sizes.index,
              cluster_sizes.values,
              color = ['#2ecc71','#3498db','#e74c3c',
                     '#f39c12','#9b59b6'])

axes[1,0].set_title('Customers per cluster')
axes[1,0].set_xlabel('Cluster')
axes[1,0].set_ylabel;("count")


#radar cgarts cluster profile
profile_norm = (profile - profile.min())/(profile.max()- profile.min())
colors = ['#2ecc71','#3498db','#e74c3c',
                     '#f39c12','#9b59b6']

for i in range(optimal_k):
    axes[1,1].plot(features, profile_norm.iloc[i],
                    'o-', color = colors[i], label= f'cluster {i}')


axes[1,1].set_title('Normalized Cluster Profiles')
axes[1,1].legend(loc='upper right', fontsize=7)
axes[1,1].tick_params(axis='x', rotation=45)

# plt.tight_layout()
# plt.show()



cluster_names = {
    0: 'Occasional Buyers',
    1: 'At Risk Customers',
    2: 'Discount Hunters',
    3: 'VIP Customers',
    4: 'Mass Market'
}

customer['Segment_name']= customer['Cluster'].map(cluster_names)


print("Cluster bussiness Profiles")
for cluster_id, name in cluster_names.items():
    group = customer[customer['Cluster'] == cluster_id]
    print(f"Cluster {cluster_id}: {name}")
    print(f"  Customers    : {len(group)}")
    print(f"  Avg Sales    : ${group['Total_Sales'].mean():,.0f}")
    print(f"  Avg Profit   : ${group['Total_Profit'].mean():,.0f}")
    print(f"  Avg Orders   : {group['Order_Count'].mean():.1f}")
    print(f"  Avg Discount : {group['Avg_Discount'].mean()*100:.1f}%")


print("top customer per cluster")    
for cluster_id, name in cluster_names.items():
    group = customer[customer['Cluster']==cluster_id]
    top = group.nlargest(1,'Total_Sales')[
        ['Customer Name','Total_Sales','Total_Profit']]
    print(f"Clusters {cluster_id} ({name}): "
          f"{top['Customer Name'].values[0]} "
          f"(${top['Total_Sales'].values[0]:,.0f})")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))


seg_revenue = customer.groupby('Segment_name')['Total_Sales'].sum()
axes[0].bar(seg_revenue.index, seg_revenue.values,
            color=['#2ecc71','#3498db','#e74c3c',
                   '#f39c12','#9b59b6'])
axes[0].set_title('Total Revenue by Segment')
axes[0].set_xlabel('Segment')
axes[0].tick_params(axis='x', rotation=45)


seg_profit = customer.groupby('Segment_name')['Total_Profit'].sum()
colors_bar = ['green' if x > 0 else 'red'
              for x in seg_profit.values]
axes[1].bar(seg_profit.index, seg_profit.values,
            color=colors_bar)
axes[1].set_title('Total Profit by Segment')
axes[1].set_xlabel('Segment')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()          
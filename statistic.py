import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df= pd.read_csv("superstore.csv", encoding= 'latin-1')

# print("statical summary")
# print(df[['Sales', 'Profit', 'Discount', 'Quantity']].describe())

# print("manual stats")
# print(f"sales mean: {df['Sales'].mean():.2f}")
# print(f"sales median: {df['Sales'].median():.2f}")
# print(f"sales std: {df['Sales'].std():.2f}")
# print(f"sales variance: {df['Sales'].var():.2f}")
# print(f"sales skewness: {df['Sales'].skew():.2f}")
# print(f"sales kurtosis: {df['Sales'].kurtosis():.2f}")

# print("mean vs median")
# print(f"sales mean: {df['Sales'].mean():.2f}")
# print(f"sales median: {df['Sales'].median():.2f}")
# print(f"difference: {df['Sales'].mean() - df['Sales'].median():.2f}")


# fig, axes = plt.subplots(2, 2, figsize=(14, 18))
# fig.suptitle('Sales, Profit, Discount, and Quantity Distributions', fontsize=16)

# axes[0,0].hist(df['Sales'], bins=50,
#             color='steelblue', edgecolor= 'white',
#             density= True)

# mu, std = df['Sales'].mean(), df['Sales'].std()   
# x= np.linspace(df['Sales'].min(), df['Sales'].max(), 100)
# axes[0,0].plot(x, stats.norm.pdf(x , mu, std), 'r-', linewidth= 2, label= 'Normal curve')

# axes[0,0].set_title('Sales distribution vs normal distribution')
# axes[0,0].legend()

# #chart 2
# axes[0,1].hist(df['Profit'], bins=50,
#                color='green', edgecolor='white')
# axes[0,1].axvline(df['Profit'].mean(), 
#                    color='red', linestyle='--', label='Mean')
# axes[0,1].axvline(df['Profit'].median(),
#                    color='blue', linestyle='--', label='Median')
# axes[0,1].set_title('Profit Distribution')
# axes[0,1].legend()

# # Chart 3 — Box plots
# df[['Sales','Profit']].boxplot(ax=axes[1,0])
# axes[1,0].set_title('Box Plot Comparison')



# # Chart 4 — QQ Plot (normality check)
# stats.probplot(df['Sales'], dist="norm", plot=axes[1,1])
# axes[1,1].set_title('Q-Q Plot (Is Sales Normal?)')


# plt.tight_layout()
# plt.show()


print(df['Sales'].isnull().sum())      # check for missing values
print(df['Sales'].dtype)                # check the column type

# Force numeric, drop any NaNs
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df = df.dropna(subset=['Sales'])
sales = df['Sales'].values
sales = df['Sales'].values


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Central Limit Theorem', fontsize=16)

axes[0].hist(sales, bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Orginal sales(skewed)')
axes[0].axvline(sales.mean(), color='red', linestyle='--', label=f'Mean={sales.mean():.0f}')
axes[0].legend()


#sample mean = 30
sample_mean_30 = [np.random.choice(sales, 30).mean() for _ in range(1000)]
axes[1].hist(sample_mean_30, bins=50, color='green', edgecolor='white')
axes[1].set_title('Sample means(n=30, 1000 samples)')
axes[1].axvline(np.mean(sample_mean_30), color='red', linestyle='--')

#sample_mean_100
# Sample means n=100
sample_means_100 = [np.random.choice(sales, 100).mean() for _ in range(1000)]
axes[2].hist(sample_means_100, bins=50, color='orange', edgecolor='white')
axes[2].set_title('Sample Means (n=100, 1000 samples)')
axes[2].axvline(np.mean(sample_means_100), color='red', linestyle='--')

plt.tight_layout()
plt.show()

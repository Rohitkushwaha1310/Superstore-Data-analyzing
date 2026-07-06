import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import poisson

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Poisson Distribution', fontsize=16)

# # Generate normal data
# np.random.seed(42)
# data = np.random.normal(loc=229.86, scale=150, size=1000)

# # Chart 1 — Normal curve
# axes[0].hist(data, bins=50, density=True,
#              color='steelblue', edgecolor='white', alpha=0.7)
# x = np.linspace(data.min(), data.max(), 100)
# axes[0].plot(x, stats.norm.pdf(x, data.mean(), data.std()),
#              'r-', linewidth=2)
# axes[0].axvline(data.mean(), color='green',
#                 linestyle='--', label='Mean')
# axes[0].axvline(data.mean() + data.std(),
#                 color='orange', linestyle='--', label='+1 STD')
# axes[0].axvline(data.mean() - data.std(),
#                 color='orange', linestyle='--', label='-1 STD')
# axes[0].set_title('Normal Distribution with STD')
# axes[0].legend()

# # Chart 2 — 68-95-99.7 rule
# axes[1].hist(data, bins=50, density=True,
#              color='steelblue', edgecolor='white', alpha=0.7)
# axes[1].plot(x, stats.norm.pdf(x, data.mean(), data.std()),
#              'r-', linewidth=2)
# axes[1].axvspan(data.mean()-data.std(),
#                 data.mean()+data.std(),
#                 alpha=0.2, color='green', label='68%')
# axes[1].axvspan(data.mean()-2*data.std(),
#                 data.mean()+2*data.std(),
#                 alpha=0.1, color='blue', label='95%')
# axes[1].set_title('68-95-99.7 Rule')
# axes[1].legend()

# plt.tight_layout()
# plt.show()

# # Z-score calculation
# print("\n=== Z-SCORE ANALYSIS ===")
# z_scores = stats.zscore(data)
# print(f"Mean Z-score: {z_scores.mean():.2f}")
# print(f"Outliers (|Z|>3): {(np.abs(z_scores)>3).sum()}")



# n= 100
# p= 0.30

# x= np.arange(0, n+1)
# pmf= stats.binom.pmf(x, n, p)

# # chrtr 1 
# axes[0].bar(x, pmf, color='steelblue', alpha=0.7)
# axes[0].axvline(n*p, color='red', linestyle='--',
#                 label=f'Expected = {n*p}')

# axes[0].set_xlim(10,50)
# axes[0].set_title('Expected Buyers from 100 customers')
# axes[0].set_xlabel('Number of buyers')
# axes[0]. legend()


# # chart 2 
# prob_35_plus = 1 - stats.binom.cdf(35, n, p)
# axes[1].bar(x, pmf, color='steelblue', alpha=0.4)
# axes[1].bar(x[35:], pmf[35:], color='red', 
#             alpha=0.7, label=f'P(>35) = {prob_35_plus:.3f}')
# axes[1].set_xlim(10, 50)
# axes[1].set_title('Probability of >35 Buyers')
# axes[1].legend()

# plt.tight_layout()
# plt.show()


# print(f"Expected buyers      : {n*p}")
# print(f"Std deviation        : {(n*p*(1-p))**0.5:.2f}")
# print(f"P(exactly 30 buyers) : {stats.binom.pmf(30, n, p):.4f}")
# print(f"P(more than 35)      : {prob_35_plus:.4f}")
# print(f"P(less than 25)      : {stats.binom.cdf(24, n, p):.4f}")

lambda_rate = 10  

x = np.arange(0, 25)
pmf = poisson.pmf(x, lambda_rate)

# Chart 1 — Poisson distribution
axes[0].bar(x, pmf, color='green', alpha=0.7)
axes[0].axvline(lambda_rate, color='red',
                linestyle='--', label=f'λ={lambda_rate}')
axes[0].set_title('Orders per Hour (λ=10)')
axes[0].set_xlabel('Number of orders')
axes[0].set_ylabel('Probability')
axes[0].legend()

# Chart 2 — Compare different lambda values
for lam, color in zip([5, 10, 20], ['blue', 'green', 'red']):
    pmf_temp = poisson.pmf(x, lam)
    axes[1].plot(x, pmf_temp, 'o-',
                 color=color, label=f'λ={lam}')
axes[1].set_title('Poisson with Different λ values')
axes[1].set_xlabel('Number of orders')
axes[1].legend()

plt.tight_layout()
plt.show()

print("=== POISSON ANALYSIS ===")
print(f"P(exactly 10 orders) : {poisson.pmf(10, lambda_rate):.4f}")
print(f"P(more than 15)      : {1-poisson.cdf(15, lambda_rate):.4f}")
print(f"P(less than 5)       : {poisson.cdf(4, lambda_rate):.4f}")
print(f"Mean = Variance      : {lambda_rate} = {lambda_rate}")




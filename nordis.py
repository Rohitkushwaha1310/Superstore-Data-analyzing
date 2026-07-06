import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import expon
from scipy.stats import poisson

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Exponential Distribution ', fontsize=16)

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

# 
from scipy.stats import expon

# Scenario:
# Average time between orders = 6 minutes
# What is probability of waiting X minutes?

scale = 6  # average time between orders

x = np.linspace(0, 40, 1000)
pdf = expon.pdf(x, scale=scale)

# Chart 1 — Exponential curve
axes[0].plot(x, pdf, 'b-', linewidth=2)
axes[0].fill_between(x, pdf, alpha=0.3, color='blue')
axes[0].axvline(scale, color='red', linestyle='--',
                label=f'Mean={scale} mins')
axes[0].set_title('Time Between Orders')
axes[0].set_xlabel('Minutes')
axes[0].legend()

# Chart 2 — Probability of waiting more than X mins
probs = [1-expon.cdf(t, scale=scale) for t in range(0, 31, 5)]
axes[1].bar(range(0, 31, 5), probs, color='orange',
            width=4, alpha=0.7)
axes[1].set_title('P(Wait > X minutes)')
axes[1].set_xlabel('Minutes')
axes[1].set_ylabel('Probability')

plt.tight_layout()
plt.show()

print("=== EXPONENTIAL ANALYSIS ===")
print(f"P(wait > 6 mins)  : {1-expon.cdf(6, scale=scale):.4f}")
print(f"P(wait > 10 mins) : {1-expon.cdf(10, scale=scale):.4f}")
print(f"P(wait < 3 mins)  : {expon.cdf(3, scale=scale):.4f}")



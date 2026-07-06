import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df= pd.read_csv("superstore.csv", encoding='Latin-1')

# print("Before cleaning:", df['Sales'].isnull().sum(), "nulls")
# df = df.dropna(subset=['Sales'])  
# # dropna() removes rows where Sales is null
# # subset=['Sales'] means only check Sales column
# print("After cleaning:", df['Sales'].isnull().sum(), "nulls")

# ---- ONE SAMPLE T-TEST ----
# ttest_1samp() compares sample mean to a known value
# Arguments:
#   a     = your data (Sales column)
#   popmean = value you're comparing against (200)
# Returns:
#   t_stat  = how many std devs away from target
#   p_value = probability result is due to chance

# t_stat, p_value = stats.ttest_1samp(df['Sales'], popmean=200)

# print("\n=== ONE SAMPLE T-TEST ===")
# print(f"Sample Mean  : {df['Sales'].mean():.2f}")
# print(f"Target Mean  : 200")
# print(f"T-statistic  : {t_stat:.4f}")
# print(f"P-value      : {p_value:.6f}")
# print(f"Alpha        : 0.05")

# if p_value < 0.05:
#     print("✅ REJECT H0 — Sales significantly different from 200!")
# else:
#     print("❌ FAIL TO REJECT H0 — No significant difference!")


# tech = df[df['Category']== 'Technology']['Sales'].dropna()  
# furniture = df[df['Category']== 'Furniture']['Sales'].dropna() 


# print("group stats")
# print(f"Technology mean : {tech.mean():.2f}")
# print(f"Furniture mean : {furniture.mean():.2f}")
# print(f"tech std:{tech.std():.2f}")
# print(f"furniture std:{furniture.std():.2f}")

# # 2 sampel test

# t_stat, p_value = stats.ttest_ind(tech, furniture, equal_var=False)

# print(" two sample test")
# print(f"T-stataistics : {t_stat:.4f}")
# print(f"p_value : {p_value:.6f}")

# if p_value < 0.05:
#     print("✅ REJECT H0 — Significant difference between groups!")
# else:
#     print("❌ FAIL TO REJECT — No significant difference!")


# plt.figure(figsize=(10,5))
# sns.boxplot(x="Category", y="Sales", data=df)
# plt.title("salles distributin by category")
# plt.show()


# #chi sqaure test
# contingency= pd.crosstab(df['Region'], df['Category'])
# print(" contingency table")
# print(contingency)


# chi_2, p_value, dof, expected = stats.chi2_contingency(contingency)

# print("chi square test")
# print(f"Chi-square statistic: {chi_2:.4f}")
# print(f"p_value: {p_value:.6f}")
# print(f"degree of freedom: {dof}")

# if p_value < 0.05:
#     print("✅ REJECT H0 — Region & Category ARE related!")
# else:
#     print("❌ FAIL TO REJECT — Region & Category independent!")
# plt.figure(figsize=(10,6))
# contingency_pct= contingency.div(contingency.sum(axis=1), axis=0)
# contingency_pct.plot(kind='bar', stacked=True, colormap='Set2', figsize=(10,6))
# plt.title("Category distribution by region")
# plt.xlabel("Region")
# plt.ylabel("Proportion")
# plt.legend(title="Category")
# plt.tight_layout()
# plt.show()



#A/B testing 
# lets take customer 500
n= 500
group_a= np.random.normal(
    loc= 200,
    scale=50,
    size=n
)

group_b = np.random.normal(
    loc=220,
    scale=55,
    size=n
)

t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

print("A/B testing result")
print(f"Group A mean: ${group_a.mean():.2f}")
print(f"Group B mean: ${group_b.mean():.2f}")
print(f"difference: ${group_b.mean()- group_a.mean():.2f}")
print(f"t-statistics : {t_stat:.4f}")
print(f"p_value: {p_value:.6f}")

if ( p_value < 0.05):
    print("b is signifancily better - launch it ")
else:
    print(" no significance difference - keep it ")


fig, axes= plt.subplots(1,2, figsize=(14,5))   
fig.suptitle("A/B discount strategies") 

axes[0].hist(group_a, bins=30, alpha=0.6,
             color='blue', label='Group A (10%)')
axes[0].hist(group_b, bins=30, alpha=0.6,
             color='orange', label='Group B (20%)')
axes[0].axvline(group_a.mean(), color='blue',
                linestyle='--')
axes[0].axvline(group_b.mean(), color='orange',
                linestyle='--')
axes[0].set_title('Sales Distribution A vs B')
axes[0].legend()

axes[1].boxplot([group_a, group_b],
                labels=['Group A\n10% discount',
                        'Group B\n20% discount'])
axes[1].set_title('Box Plot Comparison')
axes[1].set_ylabel('Sales')

plt.tight_layout()
plt.show()

uplift = (group_b.mean() - group_a.mean()) / group_a.mean() * 100
print(f"\nSales uplift    : {uplift:.2f}%")
print(f"Extra revenue   : ${(group_b.mean()-group_a.mean())*1000:.0f}")
print(f"(if 1000 customers)")

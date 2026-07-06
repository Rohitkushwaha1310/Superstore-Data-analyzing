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


#chi sqaure test
contingency= pd.crosstab(df['Region'], df['Category'])
print(" contingency table")
print(contingency)


chi_2, p_value, dof, expected = stats.chi2_contingency(contingency)

print("chi square test")
print(f"Chi-square statistic: {chi_2:.4f}")
print(f"p_value: {p_value:.6f}")
print(f"degree of freedom: {dof}")

if p_value < 0.05:
    print("✅ REJECT H0 — Region & Category ARE related!")
else:
    print("❌ FAIL TO REJECT — Region & Category independent!")
plt.figure(figsize=(10,6))
contingency_pct= contingency.div(contingency.sum(axis=1), axis=0)
contingency_pct.plot(kind='bar', stacked=True, colormap='Set2', figsize=(10,6))
plt.title("Category distribution by region")
plt.xlabel("Region")
plt.ylabel("Proportion")
plt.legend(title="Category")
plt.tight_layout()
plt.show()

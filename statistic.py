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

print("mean vs median")
print(f"sales mean: {df['Sales'].mean():.2f}")
print(f"sales median: {df['Sales'].median():.2f}")
print(f"difference: {df['Sales'].mean() - df['Sales'].median():.2f}")


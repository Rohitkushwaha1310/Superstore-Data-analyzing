import pandas as pd 
import numpy as np 
  


df= pd.read_excel("Week1_Day1_Employee_Sales.xlsx")

# print(df.shape)

# print(df.info())
# print(df.isnull().sum())
# print(df)
# print("diplicate rows" , df.duplicated().sum())
# 
# print(df)
# print(df.shape)

# print(df[df.duplicated(keep= False)][['emp_id', 'name']])
df= df.drop_duplicates()
# print(df.shape)
# # check if there any null value
# print(df[['sales_q1','sales_q2', 'sales_q3']].isnull().sum)

# now fill the null value with column mean
for col in ['sales_q1','sales_q2', 'sales_q3']:
    mean_val = round(df[col].mean(), 2)
    df[col]= df[col].fillna(mean_val)
    print(f"{col} mean used : {mean_val}")

# #again chekc  is ther any null values
# print("after filling")    
# print(df[['sales_q1','sales_q2', 'sales_q3']].isnull().sum)    


# #check is there any nul name 
# print(df['name'].isnull().sum())

# full the empty name unknown
df['name']= df['name'].fillna('unknown')
# print(df['name'])

# print("before", df['region'].unique())
df['region']= df['region'].str.strip().str.title()
# print("after", df['region'].unique())

df['region']= df['region'].fillna('unknown')

df['join_date'] = df['join_date'].replace('N/A', np.nan)

df['join_date']= pd.to_datetime(df['join_date'], errors= 'coerce')

print(df['join_date'])
print("dtype:", df['join_date'].dtype)
# print(" check if there anything null")
print(df['join_date'].unique())
print(df['join_date'].dtype)
print(df.isnull().sum())

df.to_csv("clean_eployee_data.csv", index= False)
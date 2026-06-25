import pandas as pd
import numpy as np



df= pd.read_csv("clean_employee_data.csv")
# print(df.shape)
# print(df.head())

df = df[df['region'] != 'unknown']
# print(df)
region_sales= df.groupby('region')['sales_q1'].sum()
# print(region_sales)
# print("\ntop region" , region_sales.idxmax())


dept_stata= df.groupby('department')[['sales_q1', 'sales_q2', 'sales_q3']].agg(['mean', 'min', 'max'])
# print(dept_stata)

pivot= pd.pivot_table(
    df,
    values= 'sales_q1',
    index= 'region',
    columns= 'department',
    aggfunc= 'mean',
    fill_value= 0

)
# print(pivot)

targets = pd.DataFrame({
    'emp_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    'target_q1': [12000, 15000, 13000, 11000, 14000,
                  13500, 12500, 14500, 11500, 13000, 12000]
})


merged= pd.merge(df, targets, on='emp_id', how ='left')
merged['hit_target']= merged['sales_q1']>= merged['target_q1']
# print(merged[['name', 'sales_q1', 'target_q1', 'hit_target']].to_string())

merged['total_sales']= merged['sales_q1']+ merged['sales_q2']+ merged['sales_q3']
top3= merged.nlargest(3, 'total_sales')[['name', 'department', 'region', 'total_sales']]
print(top3.to_string())
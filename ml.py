import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
le = LabelEncoder()

df= pd.read_csv("superstore.csv", encoding='Latin-1')

#THIS I SLABEL ENCODING WHIHC ACTUALLY HURT THE MODEL WHEN WE INPUT MORE VALUE 
# data cleaning
# df = df.dropna(subset=['Sales', 'Discount', 'Profit'])
# print(f"cleaned data shape: {df.shape}")


# # x(features)= input of model
# #y (target) - what we want to predict

# x= df[['Discount', 'Quantity']]
# y= df['Sales']

# print(f" feature shape :{x.shape}")
# print(f" target shape: {y.shape}")

# x_train , x_test, y_train , y_test = train_test_split(x,y,test_size= 0.2, random_state=42)


# print(f"training set : {x_train.shape}")
# print(f"Testing set:  {x_test.shape}")

# model = LinearRegression()
# model.fit(x_train, y_train)

# print("model TRAINED")
# print(f"intercept: {model.intercept_:.2f}")
# print("Coefficient")
# for feat, coef in zip(x.columns, model.coef_):
#     print(f" {feat:12}: {coef:.4f}")

# #evaluate model
# y_pred= model.predict(x_test)    
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2= r2_score(y_test, y_pred)

# print("model evaluation")
# print(f" rmse: {rmse:.2f}")
# print(f"R2: {r2:.4f}")

# #visualization 
# fig, axes= plt.subplots(1,2, figsize=(14,5))

# axes[0].scatter(y_test, y_pred, alpha=0.3, color='steelblue')
# axes[0].plot([y_test.min(), y_test.max()],
#                 [y_test.min(), y_test.max()],
#                 'r--', linewidth=2)

# axes[0].set_xlabel('Actual sales')
# axes[0].set_ylabel('Predict Sales')
# axes[0].set_title('Actual vs Pred sales')

# #residuals
# residuals = y_test - y_pred
# axes[1].hist(residuals, bins=50, color='green',
#              edgecolor='white')
# axes[1].axvline(0, color='red', linestyle='--')
# axes[1].set_title('Residuals Distribution')
# axes[1].set_xlabel('Residual(Actual- predicted)')

# # plt.tight_layout()
# # plt.show()


# df['category_encoded']=le.fit_transform(df['Category'])
# df['region_encoded']=le.fit_transform(df['Region'])
# df['segment_encoded']=le.fit_transform(df['Segment'])

# print("Category encoding:")
# for i, cat in enumerate(le.classes_):
#     print(f"  {cat} → {i}")


# # new feature set
# x_new = df[['Discount', 'Quantity','category_encoded', 'region_encoded',
#                 'segment_encoded']]
                

# y= df['Sales']

# x_train , x_test, y_train , y_test = train_test_split(
#     x_new, y, test_size =0.2, random_state=42
# )

# model2= LinearRegression()
# model2.fit(x_train, y_train)
# y_pred2= model2.predict(x_test)

# r2_new = r2_score(y_test, y_pred2)
# rmse_new = np.sqrt(mean_squared_error(y_test, y_pred2))

# print(" iproved model")
# print( f"new r2; {r2_new:.4f}")
# print(f"new rmse; {rmse_new:.4f}")
# print(f"improvement: {((r2_new- r2)/r2*100):.1f}")

# print("fuetures coefficient")
# for feat, coef in zip(x_new.columns, model2.coef_):
#     print(f"{feat:20}: {coef:.2f}")



#ONE HOT ENCODING

df = df.dropna(subset=['Sales', 'Discount', 'Quantity'])
df_encoded = pd.get_dummies(
    df[['Sales', 'Discount', 'Quantity',
        'Category', 'Region', 'Segment']],
    columns=['Category', 'Region', 'Segment'],
    drop_first=True  
)

print("Encoded columns:")
print(df_encoded.columns.tolist())
print(f"\nShape: {df_encoded.shape}")

# ---- FEATURES & TARGET ----
X = df_encoded.drop('Sales', axis=1)
y = df_encoded['Sales']

# ---- SPLIT & TRAIN ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model3 = LinearRegression()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)

# ---- EVALUATE ----
r2   = r2_score(y_test, y_pred3)
rmse = np.sqrt(mean_squared_error(y_test, y_pred3))

print("\n=== ONE HOT ENCODED MODEL ===")
print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.2f}")

# ---- FEATURE IMPORTANCE ----
importance = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model3.coef_
}).sort_values('Coefficient', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance.to_string())


# 5 folf cross validation 

model_cv = LinearRegression()

scores = cross_val_score(
           model_cv, X, y,
           cv=5,
           scoring='r2'
)

print ("5 fols cross validation ")
print(f"fold scores: {scores.round(4)}")
print(f"mean r2 : {scores.mean():.4f}")
print(f"Std Dev     : {scores.std():.4f}")
print(f"Min R²      : {scores.min():.4f}")
print(f"Max R²      : {scores.max():.4f}")




plt.figure(figsize=(8, 5))

plt.bar(range(1,6), scores, color ='steelblue',
            edgecolor='white')


plt.axhline(scores.mean(), color ='red',
             linestyle='--', label=f"mean r2={scores.mean():.3f}")

plt.xlabel('Fold')
plt.ylabel('R2 score')
plt.title('s=fold cross validation results')
plt.legend()
plt.tight_layout()
plt.show()

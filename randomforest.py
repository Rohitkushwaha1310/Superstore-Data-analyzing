import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV



df= pd.read_csv("superstore.csv")
df= df.dropna(subset=['Sales', 'Discount', 'Quantity',
                    'Category', 'Region', 'Segment',
                    ])

df_encoded = pd.get_dummies(
    df[['Sales', 'Discount', 'Quantity','Category',
            'Region', 'Segment']],
    columns = ['Category', 'Region', 'Segment'],
    drop_first=True        
)

x= df_encoded.drop('Sales', axis=1)
y= df_encoded['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# linear regression 
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2   = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))  


# radnom forest
rf= RandomForestRegressor(
    n_estimators =100,
    random_state = 42,
    n_jobs=-1
)

rf.fit(X_train , y_train)
rf_pred= rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))


# print("=" * 45)
# print(f"{'Model':<20} {'R²':>8} {'RMSE':>12}")
# print("=" * 45)
# print(f"{'Linear Regression':<20} {lr_r2:>8.4f} {lr_rmse:>12.2f}")
# print(f"{'Random Forest':<20} {rf_r2:>8.4f} {rf_rmse:>12.2f}")
# print("=" * 45)
# print(f"R² improvement  : {((rf_r2-lr_r2)/abs(lr_r2)*100):.1f}%")
# print(f"RMSE improvement: {((lr_rmse-rf_rmse)/lr_rmse*100):.1f}%")


#visualization
fig, axes = plt.subplots(1,2, figsize=(14,5)) 
axes[0].scatter(y_test, lr_pred, alpha=0.3,
                color='blue', label='Linear Regression')
axes[0].scatter(y_test, rf_pred, alpha=0.3,
                color='green', label='Random Forest')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect')
axes[0].set_xlabel('Actual Sales')
axes[0].set_ylabel('Predicted Sales')
axes[0].set_title('LR vs RF — Actual vs Predicted')
axes[0].legend()

models = ['Linear\nRegression', 'Random\nForest']
r2s    = [lr_r2, rf_r2]
colors = ['steelblue', 'green']
axes[1].bar(models, r2s, color=colors, edgecolor='white')
axes[1].set_title('R² Score Comparison')
axes[1].set_ylabel('R² Score')
for i, v in enumerate(r2s):
    axes[1].text(i, v+0.01, f'{v:.4f}',
                 ha='center', fontweight='bold')

plt.tight_layout()
# plt.show()     



#Feature importance 
immportance_df = pd.DataFrame({
    'Feature' : x.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending= False)


# print("Fetaure Importance")
# print(immportance_df.to_string())

#viualisation

plt.figure(figsize =(10,6))
sns.barplot(
    x='Importance',
    y= 'Feature',
    data = immportance_df,
    palette = 'viridis'
)

plt.title('Random forest - fetaure importance ')
plt.xlabel('Importance Score')
plt.tight_layout()
# plt.show()


#hyperparamenter grid
param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth'    : [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV explanation:
# estimator = model to tune
# param_grid = combinations to try
# cv = cross validation folds
# scoring = metric to optimize
# n_jobs = use all CPU cores

print("🔍 Searching best parameters...")
print("This may take 1-2 minutes...")

grid_search = GridSearchCV(
    estimator  = RandomForestRegressor(random_state=42),
    param_grid = param_grid,
    cv         = 3,
    scoring    = 'r2',
    n_jobs     = -1,
    verbose    = 1
)

grid_search.fit(X_train, y_train)

print("\n=== GRID SEARCH RESULTS ===")
print(f"Best parameters : {grid_search.best_params_}")
print(f"Best CV R²      : {grid_search.best_score_:.4f}")

# ---- EVALUATE BEST MODEL ----
best_model = grid_search.best_estimator_
best_pred  = best_model.predict(X_test)
best_r2    = r2_score(y_test, best_pred)
best_rmse  = np.sqrt(mean_squared_error(y_test, best_pred))

print(f"\n=== BEST MODEL RESULTS ===")
print(f"Default RF R²   : {rf_r2:.4f}")
print(f"Tuned RF R²     : {best_r2:.4f}")
print(f"Default RMSE    : {rf_rmse:.2f}")
print(f"Tuned RMSE      : {best_rmse:.2f}")
print(f"Improvement     : {((best_r2-rf_r2)/rf_r2*100):.1f}%")
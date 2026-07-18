import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
import joblib
import os

df= pd.read_csv("superstore.csv")
df= df.dropna(subset=['Sales','Profit','Discount',
                        'Quantity','Category','Region','Segment'])

df['Is_Profitable']= (df['Profit']>0).astype(int)
df['Sales_per_Qty']= df['Sales']/ df['Quantity']   
df['Discount_Impact'] = df['Sales'] * df['Discount']

num_features= ['Discount','Quantity',
                'Sales_per_Qty','Discount_Impact' ]


cat_features = ['Category', 'Region','Segment']

x = df[num_features+ cat_features]
y= df['Is_Profitable']

x_train , x_test, y_train, y_test = train_test_split(
x,y , test_size=0.2, random_state=42)





#building pipeleining


preprocessor= ColumnTransformer(transformers= [
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown ='ignore'), cat_features)
])

#full pipeline
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))

])

pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1))
])



#train the pipleline

print("Training logistic regression pipeline")
pipeline_lr.fit(x_train, y_train)
print("traininig random forest p[iplein")
pipeline_rf.fit(x_train, y_train)

lr_pred = pipeline_lr.predict(x_test)
rf_pred = pipeline_rf.predict(x_test)

print("\n=== PIPELINE RESULTS ===")
print(f"{'Model':<25} {'Accuracy':>10}")

print(f"{'Logistic Regression':<25} {accuracy_score(y_test, lr_pred)*100:>10.2f}%")
print(f"{'Random Forest':<25} {accuracy_score(y_test, rf_pred)*100:>10.2f}%")

print("RANDOM FOREST REPORT ")
print(classification_report(y_test, rf_pred,
      target_names=['Loss','Profitable']))


param_grid = {
    'classifier__n_estimators' : [50, 100, 200],
    'classifier__max_depth'    : [5, 10, None],
    'classifier__min_samples_split': [2, 5]

}

print("Searching best parameter")

grid_search = GridSearchCV(
    estimator  = pipeline_rf,
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'f1',
    n_jobs     = -1,
    verbose    = 1
)

# grid_search.fit(x_train, y_train)
# print(f"Best parameters : {grid_search.best_params_}")
# print(f"Best CV F1      : {grid_search.best_score_:.4f}")


best_pipeline = grid_search.best_estimator_
best_pred = best_pipeline.predict(x_test)

# print(f"\nBest Pipeline Accuracy : {accuracy_score(y_test, best_pred)*100:.2f}%")
# print("\nClassification Report:")
# print(classification_report(y_test, best_pred,
#       target_names=['Loss','Profitable']))

#cross validation
cv_scores = cross_val_score(
    best_pipeline, x, y,
    cv=5, scoring='f1', n_jobs=-1)

# print(f"\n=== CROSS VALIDATION ===")
# print(f"CV F1 Scores : {cv_scores.round(4)}")
# print(f"Mean F1      : {cv_scores.mean():.4f}")
# print(f"Std Dev      : {cv_scores.std():.4f}")



save_path = r'C:\Users\Rohit\Desktop\Data analyst\best_pipeline.pkl'
joblib.dump(best_pipeline, save_path)
print(f"pipeline saved to : {save_path}")
print(f" file size : {os.path.getsize(save_path)/1024:.1f} KB")

loaded_pipeline = joblib.load(save_path)
print(f" pipeline loaded successfully")

new_orders = pd.DataFrame({
    'Discount'      : [0.0, 0.5, 0.2, 0.8, 0.1],
    'Quantity'      : [3,   1,   5,   2,   8  ],
    'Sales_per_Qty' : [250, 100, 180, 50, 300 ],
    'Discount_Impact':[0,  500, 360, 800, 300 ],
    'Category'      : ['Technology','Furniture',
                       'Office Supplies','Furniture',
                       'Technology'],
    'Region'        : ['West','South','East',
                       'Central','West'],
    'Segment'       : ['Consumer','Consumer',
                       'Corporate','Consumer',
                       'Home Office']
})

predictions = loaded_pipeline.predict(new_orders)
probabilities = loaded_pipeline.predict_proba(new_orders)[:,1]

print("production predictaions")

result = pd.DataFrame({
    'Order'      : range(1, 6),
    'Discount'   : new_orders['Discount'].values,
    'Category'   : new_orders['Category'].values,
    'Prediction' : ['✅ Profit' if p==1
                    else '❌ Loss' for p in predictions],
    'Confidence' : [f"{p*100:.1f}%" for p in probabilities]
})
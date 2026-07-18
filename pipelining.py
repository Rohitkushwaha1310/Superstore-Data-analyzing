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
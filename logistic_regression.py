import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("superstore.csv")
df= df.dropna(subset=['Sales','Profit','Discount','Quantity',
                        'Category','Region','Segment'])

df['Is_Profitable']= (df['Profit']>0).astype(int)
print("target distribution")
print(df['Is_Profitable'].value_counts())
print(f"Profitable orders : {df['Is_Profitable'].mean()*100:.1f}%")


df_ml = pd.get_dummies(
     df[['Discount','Quantity','Category',
        'Region','Segment','Is_Profitable']],
    columns=['Category','Region','Segment'],
    drop_first=True
)

x= df_ml.drop('Is_Profitable', axis=1)
y= df_ml['Is_Profitable']

x_train, x_test, y_train, y_test= train_test_split(
    x, y, test_size=0.2, random_state=42
)


lr= LogisticRegression(max_iter=1000, random_state=42)
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print(f"Accuracy : {accuracy_score(y_test, lr_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred,
      target_names=['Loss','Profitable']))


# confusion matrix
cm= confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Loss', 'Profitable'],
            yticklabels=['Loss', 'Profitable'])



plt.title('Confusion matrix- logistic regrssion')
plt.ylabel('Actual')
plt.xlabel('Predicted')
# plt.tight_layout()
# plt.show()                  

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1

    
)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_proba =  rf.predict_proba(x_test)[:,1]

print("random forest result")
print(f"Accuracy : {accuracy_score(y_test, rf_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred,
      target_names=['Loss','Profitable']))

#comparision 
print("model comparision")
print(f"{'Model':<25} {'Accuracy':>10}")

print(f"{'Logistic Regression':<25} {accuracy_score(y_test, lr_pred)*100:>10.2f}%")
print(f"{'Random Forest':<25} {accuracy_score(y_test, rf_pred)*100:>10.2f}%")

#roc curve
lr_proba = lr.predict_proba(x_test)[:,1]
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, lr_proba)
lr_auc = auc(lr_fpr,lr_tpr)
rf_auc = auc(rf_tpr, rf_tpr)


fig, axes = plt.subplots(1,2 ,figsize=(14,5))

axes[0].plot(lr_fpr, lr_tpr, 'b--',
                label = f'Logistic Reg (AUC={lr_auc:.3f})')

axes[0].plot(rf_fpr, rf_tpr, 'g-',
             label=f'Random Forest (AUC={rf_auc:.3f})')

axes[0].plot([0,1],[0,1],'r--', label='Random (AUC=0.5)')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison')
axes[0].legend()


#fetaure importance
importance = pd.DataFrame({
    'Feature'  : x.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(8)

axes[1].barh(importance['Feature'],
             importance['Importance'],
             color='steelblue')
axes[1].set_title('Top 8 Feature Importances')
axes[1].set_xlabel('Importance')
plt.tight_layout()
plt.show()

print(f"\nLogistic Regression AUC : {lr_auc:.4f}")
print(f"Random Forest AUC       : {rf_auc:.4f}")


new_orders = pd.DataFrame({
     'Discount' : [0.0,  0.5,  0.2,  0.8,  0.1],
    'Quantity' : [3,    1,    5,    2,    8  ],
    'Category_Office Supplies': [0, 0, 1, 0, 1],
    'Category_Technology'     : [1, 0, 0, 0, 0],
    'Region_East'  : [1, 0, 0, 0, 1],
    'Region_South' : [0, 1, 0, 0, 0],
    'Region_West'  : [0, 0, 1, 0, 0],
    'Segment_Consumer'      : [1, 0, 1, 0, 1],
    'Segment_Home Office'   : [0, 1, 0, 0, 0],
})

new_orders= new_orders.reindex(
    columns=x.columns, fill_value=0
)

predictions = lr.predict(new_orders)
probabilities = lr.predict_proba(new_orders)[:,1]

print(" new order preediction")
results= pd.DataFrame({
    'Order'      : range(1, 6),
    'Discount'   : [0.0, 0.5, 0.2, 0.8, 0.1],
    'Quantity'   : [3, 1, 5, 2, 8],
    'Prediction' : ['✅ Profitable' if p==1 
                    else '❌ Loss' for p in predictions],
    'Confidence' : [f"{p*100:.1f}%" 
                    for p in probabilities]
})

print(results.to_string(index=False))
print("\n BUSINESS RECOMMENDATION ")
for i, (pred, prob, disc) in enumerate(
        zip(predictions, probabilities,
            [0.0, 0.5, 0.2, 0.8, 0.1]), 1):
    if pred == 0:
        print(f"Order {i}: ⚠️ BLOCK — predicted loss! "
              f"Discount={disc*100:.0f}%")
    elif prob < 0.7:
        print(f"Order {i}: ⚠️ REVIEW — low confidence "
              f"({prob*100:.1f}%)")
    else:
        print(f"Order {i}: ✅ APPROVE — "
              f"high confidence ({prob*100:.1f}%)")
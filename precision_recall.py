import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (precision_recall_curve,
                              average_precision_score,
                              precision_score, recall_score,
                              f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("superstore.csv")
df= df.dropna(subset=['Sales','Profit','Discount',
                        'Quantity','Category','Region','Segment'])

df['Is_Profitable']= (df['Profit']>0).astype(int)
df_ml = pd.get_dummies(
    df[['Discount','Quantity','Category',
        'Region','Segment','Is_Profitable']],
    columns=['Category','Region','Segment'],
    drop_first=True)      


x= df_ml.drop('Is_Profitable', axis=1)
y= df_ml['Is_Profitable']    

x_train, x_test, y_train , y_test= train_test_split(
    x,y, test_size=0.2, random_state=42
)


lr= LogisticRegression(max_iter=1000, random_state=42)
rf= RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

lr.fit(x_train, y_train)
rf.fit(x_train, y_train)

lr_proba = lr.predict_proba(x_test)[:, 1]
rf_proba= rf.predict_proba(x_test)[:,1]

#precisionn recall curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_proba)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_proba)
lr_ap = average_precision_score(y_test, lr_proba)
rf_ap = average_precision_score(y_test, rf_proba)

axes[0].plot(lr_recall, lr_precision, 'b-',
             label=f'Logistic Reg (AP={lr_ap:.3f})')
axes[0].plot(rf_recall, rf_precision, 'g-',
             label=f'Random Forest (AP={rf_ap:.3f})')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve')
axes[0].legend()


# matrices comparision
metrics = {
    'Precision': [precision_score(y_test, lr.predict(x_test)),
                  precision_score(y_test, rf.predict(x_test))],
    'Recall'   : [recall_score(y_test, lr.predict(x_test)),
                  recall_score(y_test, rf.predict(x_test))],
    'F1 Score' : [f1_score(y_test, lr.predict(x_test)),
                  f1_score(y_test, rf.predict(x_test))]
}

x = np.arange(3)
width = 0.35
axes[1].bar(x - width/2,
            [v[0] for v in metrics.values()],
            width, label='Logistic Reg', color='steelblue')
axes[1].bar(x + width/2,
            [v[1] for v in metrics.values()],
            width, label='Random Forest', color='green')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics.keys())
axes[1].set_title('Metrics Comparison')
axes[1].set_ylim(0, 1.1)
axes[1].legend()

plt.tight_layout()
# plt.show()

# print("=== DETAILED METRICS ===")
# print(f"{'Metric':<15} {'Log Reg':>10} {'Rand Forest':>12}")
# print("-" * 40)
# for metric, (lr_val, rf_val) in metrics.items():
#      print(f"{metric:<15} {lr_val:>10.4f} {rf_val:>12.4f}")






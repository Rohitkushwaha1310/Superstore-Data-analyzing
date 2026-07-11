import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.tight_layout()
plt.show()                  


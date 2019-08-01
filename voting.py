
import pandas as pd

data = pd.read_csv("dataset.csv")

y = data["target"].values

feature_columns = []
for col in data.columns:
    if col not in ["id", "target"]:
        feature_columns.append(col)

X = data[feature_columns].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
#X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.4)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True)

model = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], 
    voting='soft'
) 
model.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(model, X_train, y_train, cv=3)
y_test_pred = cross_val_predict(model, X_test, y_test, cv=3)

from sklearn.metrics import confusion_matrix, precision_score, recall_score
con_mx = confusion_matrix(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)

y_probas = cross_val_predict(model, X_test, y_test, cv=3, method="predict_proba")
y_scores = y_probas[:, 1]

from sklearn.metrics import precision_recall_curve, roc_auc_score
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

print(model.__class__.__name__)
print(f"Precision: {round(precision, 2)}  Recall: {round(recall, 2)}  AUC: {round(roc_auc, 2)}")

import matplotlib.pyplot as plt
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
    plt.xlabel("Threshold") 
    plt.legend(loc="upper left") 
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds) 
plt.show()


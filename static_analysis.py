import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble as ek
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("https://raw.githubusercontent.com/Naveensiuuu/MalwareDetection/main/MalwareData.csv",sep='|' )

#feature selection

# Feature
X = data.drop(['Name','md5','legitimate'],axis=1).values    #Droping this because classification model will not accept object type elements (float and int only)
# Target variable
y = data['legitimate'].values
extratrees = ek.ExtraTreesClassifier().fit(X,y)
model = SelectFromModel(extratrees, prefit=True)
X_new = model.transform(X)
nbfeatures = X_new.shape[1]
#splitting
X_train, X_test, y_train, y_test = train_test_split(X_new, y ,test_size=0.29, stratify = y)
#scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#model
best_model = RandomForestClassifier(n_estimators=50)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Confusion Matrix:\n{cm}")

joblib.dump(best_model, 'classifier.pkl')

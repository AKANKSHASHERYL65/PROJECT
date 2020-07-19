
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc

dataset = pd.read_csv('creditcard.csv')

dataset.head()

dataset.isna()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset['normalizedAmount'] = sc.fit_transform(dataset['Amount'].values.reshape(-1,1))
dataset = dataset.drop(['Amount'], axis = 1)

dataset.head()

dataset = dataset.drop(['Time'], axis = 1)
dataset.head()

X = dataset.iloc[:, dataset.columns != 'Class'].values
y = dataset.iloc[:, dataset.columns == 'Class'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state = 0)

X_train.shape


X_test.shape

from sklearn.ensemble import RandomForestClassifier


random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True)

print(cm)

sns.countplot(x = 'Class', data=dataset)

sns.countplot(x = 'Class', data=dataset)

from imblearn.over_sampling import SMOTE

X_train, y_train = SMOTE().fit_sample(X_train, y_train)

y_test = np.concatenate(y_test)

sns.countplot(y_test)

random_forest_resampled = RandomForestClassifier(n_estimators=100)

random_forest_resampled.fit(X_train, y_train)


y_pred = random_forest_resampled.predict(X_test)

cm = confusion_matrix(y_test, y_pred)


sns.heatmap(cm, annot = True)

print(cm)

from sklearn.metrics import classification_report

classification_report(y_test, y_pred)

print(classification_report(y_test, y_pred))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


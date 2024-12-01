import seaborn as sns
import numpy as np

from io import BytesIO
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)

print('Accuracy:', np.round(metrics.accuracy_score(y_test, ypred), 4))
print('Precision:', np.round(metrics.precision_score(y_test, ypred, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(y_test, ypred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(y_test, ypred, average='weighted'), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test, ypred), 4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(y_test, ypred), 4))
print('\t\tClassification Report:\n', metrics.classification_report(ypred, y_test))

mat = confusion_matrix(y_test, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.savefig("Confusion.jpg")
f = BytesIO()
plt.savefig(f, format="svg")

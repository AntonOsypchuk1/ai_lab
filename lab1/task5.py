import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilities import find_TP, find_FN, find_FP, find_TN, osypchuk_confusion_matrix, osypchuk_accuracy_score, osypchuk_recall_score, osypchuk_precision_score, osypchuk_f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data/data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

# Check on confusion matrix values
print('TP:',find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:',find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:',find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:',find_TN(df.actual_label.values, df.predicted_RF.values))

# Assert for osypchuk_confusion_matrix
confusion_matrix(df.actual_label.values, df.predicted_RF.values)

assert np.array_equal(
    osypchuk_confusion_matrix(df.actual_label.values, df.predicted_RF.values), 
    confusion_matrix(df.actual_label.values, df.predicted_RF.values)), 'osypchuk_confusion_matrix() is not correct for RF'
assert np.array_equal(
    osypchuk_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
    confusion_matrix(df.actual_label.values, df.predicted_LR.values)), 'osypchuk_confusion_matrix() is not correct for LR'

# Assert for osypchuk_accuracy_score
accuracy_score(df.actual_label.values, df.predicted_RF.values)

assert osypchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values),'osypchuk_accuracy_score failed on RF'
assert osypchuk_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'osypchuk_accuracy_score failed on LR'

print('Accuracy RF: %.3f'%(osypchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f'%(osypchuk_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

# Assert for osypchuk_recall_score
recall_score(df.actual_label.values, df.predicted_RF.values)

assert osypchuk_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values), 'osypchuk_recall_score failed on RF'
assert osypchuk_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values), 'osypchuk_recall_score failed on LR'
print('Recall RF: %.3f'%(osypchuk_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f'%(osypchuk_recall_score(df.actual_label.values, df.predicted_LR.values)))

# Assert for osypchuk_precision_score
precision_score(df.actual_label.values, df.predicted_RF.values)

assert osypchuk_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values), 'osypchuk_precision_score failed on RF'
assert osypchuk_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values), 'osypchuk_precision_score failed on LR'
print('Precision RF: %.3f'%(osypchuk_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f'%(osypchuk_precision_score(df.actual_label.values, df.predicted_LR.values)))

# Assert for osypchuk_f1_score
f1_score(df.actual_label.values, df.predicted_RF.values)

assert osypchuk_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values), 'osypchuk_f1_score failed on RF'
assert osypchuk_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values), 'osypchuk_f1_score failed on LR'
print('F1 RF: %.3f'%(osypchuk_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f'%(osypchuk_f1_score(df.actual_label.values, df.predicted_LR.values)))

print('scores with threshold = 0.5')
print('Accuracy RF: %.3f'%(osypchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f'%(osypchuk_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f'%(osypchuk_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f'%(osypchuk_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25') 
print('Accuracy RF: %.3f'%(osypchuk_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f'%(osypchuk_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f'%(osypchuk_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f'%(osypchuk_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

# roc_curve & roc_auc_score
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

# Plot ROC curve
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# AUC
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

# Plot ROC curve with AUC
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
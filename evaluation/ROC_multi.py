# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import from_file_get_list as getfile

# 将标签二值化
y = getfile.get_test_labels('./labels.txt')
y_pre = getfile.get_predicted_labels('./model/predicted.txt')
y_pre = label_binarize(y_pre, classes=[0, 1, 2, 3, 4, 5, 6])
y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6])
# 设置种类
n_classes = 7

y_score = np.load('./presents.npy')

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i], pos_label=y_pre[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(i)

# Plot all ROC curves
lw=2
plt.figure()

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import from_file_get_list as getfile

# 89.4--25  89.5--12  88.3--10  90.3--32  81.0--1  81.5--3  57.2--33

y_test = getfile.get_test_labels('./model/model--87/labels24-87.txt')
y_score_raw = np.load('./model/model--87/presents24.npy')
y_predicted = getfile.get_test_labels('./model/model--87/predicted24-87.txt')
y_score_new = []
for i in y_score_raw:
    y_score_new.append(np.max(i))

# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y_test, y_score_new, pos_label=y_predicted) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) # 计算auc的值

lw = 2
plt.figure(figsize=(6*1.2,6))
plt.plot(fpr, tpr, color='red',
         lw=lw, label='AUC = %0.2f' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC plot')
plt.legend(loc="lower right")
plt.savefig('./model/acc-87/evaluation_and_plot/ROC_plot.pdf')
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# from sklearn import model_selection
#
# # Import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# ##变为2分类
# X, y = X[y != 2], y[y != 2]
#
# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#
# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.3,random_state=0)
#
# # Learn to predict each class against the other
# svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)
#
# ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
# y_score = svm.fit(X_train, y_train).decision_function(X_test)
# print(y_score)
#
# # Compute ROC curve and ROC area for each class
# fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
# roc_auc = auc(fpr,tpr) ###计算auc的值
#
# lw = 2
# plt.figure(figsize=(6,6))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


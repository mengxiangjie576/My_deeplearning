# encoding:utf-8
import sklearn.metrics

from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from itertools import cycle
import random
import os
from time import time


def PCA_plot(data, label):
    pca = PCA().fit_transform(data)
    # 设置画布大小
    plt.figure(figsize=(8, 8))
    # plt.subplot(122)
    plt.scatter(pca[:, 0], pca[:, 1], c=label)
    plt.colorbar()  # 使用这一句就可以分辨出，颜色对应的类了！神奇啊。
    plt.savefig('./model/evaluation_and_plot/pca.jpg')
    plt.show()



















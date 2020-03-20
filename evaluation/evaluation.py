# encoding:utf-8

import torch
import sklearn.preprocessing
import numpy as np
import os


import from_file_get_list as getfile
import evaluation_func_and_plot as ev
from ResNet import ResNet18


n_classes = 7

# 定义是否使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_one_hot(one_list):
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit([[0], [1], [2], [3]])
    return enc.transform(one_list).toarray()


def get_raw_list(one_list):
    """将一般列表变为适合get_one_hot函数的列表"""
    new_list = []
    for i in one_list:
        j = []
        j.append(i)
        new_list.append(j)
    return new_list

    # # 绘制PCA图
    # if os.path.exists('./model/evaluation_and_plot/pca.jpg') is not True:
    #     features = np.load('./model/new_features.npy')
    #     new_features = []
    #     labels = getfile.get_test_labels('./model/labels.txt')
    #     cl, cl_i = get_classes(labels, 100, 7)
    #     for i in cl_i:
    #         new_features.append(features[i])
    #     print(len(cl))
    #     print(len(new_features))
    #
    #     data = np.reshape(new_features, (len(new_features), -1))
    #     ev.PCA_plot(data, cl)




























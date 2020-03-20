import from_file_get_list as getfile
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class_num = 2


def get_classes(labels, num, class_num):
    cl = []
    cl_i = []
    for i in range(class_num):
        x = 0
        for en, j in enumerate(labels):
            if i == j and x <= num:
                cl.append(j)
                cl_i.append(en)
                x += 1
    return cl, cl_i


def t_sne_plot(plot_path, features, labels, classes):
    # 使用TSNE进行降维处理。从100维降至2维。
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(features)
    tsnex0 = []
    tsney0 = []
    tsnex1 = []
    tsney1 = []
    tsnex2 = []
    tsney2 = []
    tsnex3 = []
    tsney3 = []
    tsnex4 = []
    tsney4 = []
    tsnex5 = []
    tsney5 = []
    tsnex6 = []
    tsney6 = []
    for index, i in enumerate(labels):
        if i == 0:
            tsnex0.append(tsne[index, 0])
            tsney0.append(tsne[index, 1])
    for index, i in enumerate(labels):
        if i == 1:
            tsnex1.append(tsne[index, 0])
            tsney1.append(tsne[index, 1])
    # for index, i in enumerate(labels):
    #     if i == 2:
    #         tsnex2.append(tsne[index, 0])
    #         tsney2.append(tsne[index, 1])
    # for index, i in enumerate(labels):
    #     if i == 3:
    #         tsnex3.append(tsne[index, 0])
    #         tsney3.append(tsne[index, 1])
    # for index, i in enumerate(labels):
    #     if i == 4:
    #         tsnex4.append(tsne[index, 0])
    #         tsney4.append(tsne[index, 1])
    # for index, i in enumerate(labels):
    #     if i == 5:
    #         tsnex5.append(tsne[index, 0])
    #         tsney5.append(tsne[index, 1])
    # for index, i in enumerate(labels):
    #     if i == 6:
    #         tsnex6.append(tsne[index, 0])
    #         tsney6.append(tsne[index, 1])

    # 设置画布大小
    plt.figure(figsize=(7, 6))
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=labels)
    plt.scatter(tsnex0[:], tsney0[:], label='Normal', c='orange', s=12)
    plt.scatter(tsnex1[:], tsney1[:], label='Abnormal', c='navy', s=12)
    # plt.scatter(tsnex2[:], tsney2[:], label=classes[2], c='b', s=10)
    # plt.scatter(tsnex3[:], tsney3[:], label=classes[3], c='y', s=10)
    # plt.scatter(tsnex4[:], tsney4[:], label=classes[4], c='m', s=10)
    # plt.scatter(tsnex5[:], tsney5[:], label=classes[5], c='c', s=10)
    # plt.scatter(tsnex6[:], tsney6[:], label=classes[6], c='k', s=10)

    plt.legend(loc="upper right")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Features from ResNet-18(Test)')

    plt.savefig(plot_path)
    plt.show()


classes = ['benign', 'malignant']
features = np.load('./model/model--87/new_features24.npy')
new_features = []
labels = getfile.get_test_labels('./model/model--87/labels24-87.txt')
cl, cl_i = get_classes(labels, 300, class_num)
for i in cl_i:
    new_features.append(features[i])
data = np.reshape(new_features, (len(new_features), -1))
t_sne_plot('./model/acc-87/evaluation_and_plot/tsne_plot.pdf', data, cl, classes)




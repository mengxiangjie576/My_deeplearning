import from_file_get_list as getfile
import sklearn
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


def confusion_matrix(tests, pres, path):
    confusion_matrix = sklearn.metrics.confusion_matrix(tests, pres)
    print(confusion_matrix)
    plt.matshow(confusion_matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)
    plt.show()
    return confusion_matrix


labels = getfile.get_test_labels(r'./model/model--87/labels24-87.txt')
predicted = getfile.get_predicted_labels(r'./model/model--87/predicted24-87.txt')
predicteds = []
cl, cl_i = get_classes(labels, 100, class_num)
for i in cl_i:
    predicteds.append(predicted[i])
cm = confusion_matrix(cl, predicteds, r'.\model/acc-87/evaluation_and_plot/confusion_matrix.pdf')
f = open(r'./model/acc-87/evaluation_and_plot/confusion_matrix.txt', 'a')
for line in cm:
    f.write(str(line) + '\n')
f.close()






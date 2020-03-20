import from_file_get_list as getfile
import sklearn.metrics

mode = 0

if mode == 0:  # 多分类
    labels = getfile.get_test_labels('./model/model--87/labels24-87.txt')
    predicted = getfile.get_predicted_labels('./model/model--87/predicted24-87.txt')
    target_names = ['benign', 'malignant']
    # 调用接口
    report = sklearn.metrics.classification_report(labels, predicted, target_names=target_names, digits=4)
    f = open('./model/acc-87/evaluation_and_plot/evaluation_index.txt', 'a')
    print(report)
    f.write(report)
    f.close()

if mode == 1:  # 二分类
    def evaluation_index(labels, pres):
        """
        计算二分类的评价指标
        :param labels: 实际的label列表
        :param pres: 预测的label列表
        """
        recall = sklearn.metrics.recall_score(labels, pres, average='micro')
        precision = sklearn.metrics.precision_score(labels, pres, average='micro')
        f1_score = sklearn.metrics.f1_score(labels, pres, average='micro')
        accuracy = sklearn.metrics.accuracy_score(labels, pres)

        return recall, precision, f1_score, accuracy

    labels = getfile.get_test_labels('./model/model--87/labels24-87.txt')
    predicted = getfile.get_predicted_labels('./model/model--87/predicted24-87.txt')
    recall, precision, f1_score, accuracy = evaluation_index(labels, predicted)
    f = open('./model/acc-87/evaluation_and_plot/evaluation_index.txt', 'a')
    f.write('recall:' + str(recall) + '\n')
    f.write('precision:' + str(precision) + '\n')
    f.write('f1 score:' + str(f1_score) + '\n')
    f.write('accuracy:' + str(accuracy) + '\n')





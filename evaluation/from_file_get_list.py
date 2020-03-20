# encoding:utf-8
import re


def get_loss(file_path):
    loss = []
    f = open(file_path, 'r')
    for line in f:
        loss.append(float(line))
    return loss


def get_acc(file_path):
    acc = []
    f = open(file_path, 'r')
    for line in f:
        acc.append(float(line))
    return acc


def get_test_labels(file_path):
    test_labels = []
    f = open(file_path, 'r')
    for line in f:
        new_line = re.findall("\d+", line)[0]
        test_labels.append(int(new_line))
    f.close()
    return test_labels


def get_predicted_labels(file_path):
    predicted_labels = []
    f = open(file_path, 'r')
    for line in f:
        new_line = re.findall("\d+", line)[0]
        predicted_labels.append(int(new_line))
    f.close()
    return predicted_labels


def get_present(file_path):  # ???????????????????????????????????????????????
    present = []
    f = open(file_path, 'r')
    for line in f:
        new_line = re.findall("\.+", line)[0]
        present.append(float(new_line))
    f.close()
    return present


if __name__ == '__main__':
    predicted = get_predicted_labels('./model/present.txt')
    print(predicted)






































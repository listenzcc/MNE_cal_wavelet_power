# code: utf-8

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.svm import SVC
from tools import para_setting, load_file

# para init
file_dir = os.path.join('D:/', 'BeidaShuju', 'rawdata')
subject_name = 'ZYF'
fname_list, ortids, event_id, tmin, t0, tmax = para_setting(
    os.path.join(file_dir, subject_name))
dir_save = 'data_storage_'

# load data from file
print('\n'.join(fname_list))
epochs_all = []
for j, fname in enumerate(fname_list):
    print(fname)
    epochs = load_file(os.path.join(dir_save, '_'.join(
        ['epochs', subject_name, os.path.basename(fname)])+'.pkl'))
    epochs_all.append(epochs)


def zscore_on_baseline(data):
    # assert data.shape == (12, 306, 1001)
    # make sure of it
    # remove baseline using -150(50) ~ -50(150)ms
    # return baseline removed data
    mean = np.mean(data[:, :, 50:150])
    std = np.std(data[:, :, 50:150])
    data -= mean
    data /= std
    return data


# data_collection storages data from mne struct
data_collection = dict()
for ort in event_id:
    print(ort)
    data_collection[ort] = np.concatenate(tuple(
        zscore_on_baseline(epochs_all[e][ort].get_data())
        for e in range(5)), axis=0)


def random_fetch(data_col):
    # make sure data_col is deepcopy of data_collection
    # seperate training and testing data randomly
    # mean each 12 trails to make supertrail as sample
    # return training / testing, data / label

    # shuffle trails in each ort, and mean each 12 trails
    mean_sample = dict()
    label_mean_sample = dict()
    for j, ort in enumerate(event_id):
        # print('%d:' % j, ort)
        data = data_col[ort]
        np.random.shuffle(data)
        mean_sample[j] = np.concatenate(
            tuple(np.mean(data[(j*12):(j*12+12)], axis=0, keepdims=True)
                  for j in range(5)))
        # print(mean_sample[j].shape)
        label_mean_sample[j] = np.zeros(5) + j+1

    # seperate training and testing data
    train_data = np.concatenate(
        tuple(mean_sample[j][0:4] for j in range(5)))
    train_label = np.concatenate(
        tuple(label_mean_sample[j][0:4] for j in range(5)))
    test_data = np.concatenate(
        tuple(mean_sample[j][4:5] for j in range(5)))
    test_label = np.concatenate(
        tuple(label_mean_sample[j][4:5] for j in range(5)))

    return train_data, train_label, test_data, test_label


clf = SVC(gamma='auto')


def train_and_test(X, y, X_, y_, clf=clf):
    # calculate accuracy on obtained dataset
    # X: train_data
    # y: train_label
    # X_: test_data
    # y_: test_label
    # return accuracy

    # train clf
    clf.fit(X, y)
    # test clf
    check = y_ == clf.predict(X_)
    # calculate accuracy and return
    acc = sum(check) / len(check)
    return acc


time_point = 1001
repeat = 100

accuracy_all = np.zeros([repeat, time_point])
for rep in range(repeat):
    print('%d:' % rep)
    train_data, train_label, test_data, test_label = random_fetch(
        deepcopy(data_collection))
    y, y_ = train_label, test_label
    for tp in range(time_point):
        X, X_ = train_data[:, :, tp], test_data[:, :, tp]
        accuracy_all[rep, tp] = train_and_test(X, y, X_, y_)

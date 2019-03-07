# code: utf-8

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tools import para_setting, load_file

# para init
file_dir = os.path.join('D:/', 'BeidaShuju', 'rawdata')
subject_name = 'ZYF'
fname_list, ortids, event_id, tmin, t0, tmax = para_setting(
    os.path.join(file_dir, subject_name))
dir_save = 'data_storage_'

data_collection_run_epochs = dict()
for fname in fname_list:
    print(fname)
    data_collection_run_epochs[fname] = dict()
    for ort in event_id.keys():
        power_epochs = load_file(os.path.join(dir_save, '_'.join(
            ['power_epochs', subject_name, os.path.basename(fname), ort]) +
            '.pkl'))
        data_collection_run_epochs[fname][ort] = np.concatenate(tuple(
            power_epochs[j].data[np.newaxis, :, :, :]
            for j in range(len(power_epochs))))

data_collection_epochs = dict()
for ort in event_id.keys():
    print(ort)
    data_collection_epochs[ort] = np.concatenate(tuple(
        data_collection_run_epochs[fn][ort] for fn in fname_list))


def random_fetch(data_col):
    # make sure data_col is deepcopy of data_collection
    # seperate training and testing data randomly
    # data_col[ort] is numpy.array 60 x 306 x 20 x 1001
    # !!! 20: frequencies, first mean this dimension
    # mean each 12 trails to make supertrail as sample
    # return training / testing, data / label

    # shuffle trails in each ort, and mean each 12 trails
    mean_sample = dict()
    label_mean_sample = dict()
    for j, ort in enumerate(event_id):
        # print('%d:' % j, ort)
        data = data_col[ort]
        data = np.nanmean(data, axis=2)
        np.random.shuffle(data)
        print(len(data))
        mean_sample[j] = np.concatenate(
            tuple(np.nanmean(data[(j*12):(j*12+12)], axis=0, keepdims=True)
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


clf = Pipeline((("scaler", StandardScaler()),
                ("svc", SVC(gamma='auto'))))


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

accuracy_epochs = np.zeros([repeat, time_point])
for rep in range(repeat):
    print('%d:' % rep)
    train_data, train_label, test_data, test_label = random_fetch(
        deepcopy(data_collection_epochs))
    y, y_ = train_label, test_label
    for tp in range(time_point):
        X, X_ = train_data[:, :, tp], test_data[:, :, tp]
        accuracy_epochs[rep, tp] = train_and_test(X, y, X_, y_)

path_on_save = os.path.join('result_storage_', 'freqdomain')
if not os.path.exists(path_on_save):
    os.mkdir(path_on_save)
np.save(os.path.join(path_on_save, 'accuracy_epochs.npy'),
        accuracy_epochs)

plt.plot(np.mean(accuracy_epochs, axis=0))
plt.show()

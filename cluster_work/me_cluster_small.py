# -*- coding:utf-8 _*-
from scipy.ndimage import zoom
from multiprocessing import Pool
import gc
from typing import List
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn import metrics
import shutil
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from functools import reduce
from tqdm import tqdm
import multiprocessing as mp
import radiomics.featureextractor as featureextractor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy
import pandas as pd
import torch
from sklearn.cluster import KMeans
import pywt
import sys
from labelme.utils import *
import json
import cv2
import numpy as np
import os
import SimpleITK as sitk
import os
from numba import jit

# import
NUM_CLASSES = 4
NUM_SHAPE_CLUSTER = 4
NUM_CLUSTER = 4
TEST_SIZE = -1
TRAIN_DIR = './weights/me_small_test'
LOGGER_PATH = f'{TRAIN_DIR}/train_cluster.txt'
FULL_WIDTH = 384
FULL_HEIGHT = 768
params = './CT3D.yaml'


def logging_info(x):
    with open(LOGGER_PATH, "a") as file:
        file.write(x+'\n')
    print(x)


STATIC_COLORS = np.array([
    np.array([0, 0, 0]),
    np.array([255, 255, 0]),
    np.array([255, 0, 0]),  # face
    np.array([0, 0, 255]),  # lb
    np.array([0, 255, 0]),   # rb
    np.array([0, 255, 255]),   # le
])

# prepocess after cluster

SUM_A = 0.0
SUM_M = 0.0
SUM_H = 0.0
SUM_C = 0.0
SUM_V = 0.0
SUM_RMSE, SUM_R2 = 0.0, 0.0


def check_metric(y_true, y_pred, name, ans_num):
    class_ans = np.argmax(ans_num, axis=1)
    for i in range(len(y_pred)):
        y_pred[i] = class_ans[y_pred[i]]
    print(np.unique(y_pred), np.unique(y_true))
    # 计算聚类度量指标
    for cls_t in np.unique(y_true):
        y_pred_item, y_true_item = (y_pred == cls_t)*1, (y_true == cls_t)*1
        ari = metrics.adjusted_rand_score(y_pred_item, y_true_item)
        mi = metrics.adjusted_mutual_info_score(y_pred_item, y_true_item)
        homogeneity = metrics.homogeneity_score(y_pred_item, y_true_item)
        completeness = metrics.completeness_score(y_pred_item, y_true_item)
        v_measure = metrics.v_measure_score(y_pred_item, y_true_item)
        logging_info(
            f"for cls {name} classes {cls_t}, the Adjusted Rand Index (ARI): {ari}")
        logging_info(
            f"for cls {name} classes {cls_t}, the Adjusted Mutual Information (MI): {mi}")
        logging_info(
            f"for cls {name} classes {cls_t}, the Homogeneity: {homogeneity}")
        logging_info(
            f"for cls {name} classes {cls_t}, the Completeness: {completeness}")
        logging_info(
            f"for cls {name} classes {cls_t}, the V-measure: {v_measure}")
    ari = metrics.adjusted_rand_score(y_pred, y_true)
    mi = metrics.adjusted_mutual_info_score(y_pred, y_true)
    homogeneity = metrics.homogeneity_score(y_pred, y_true)
    completeness = metrics.completeness_score(y_pred, y_true)
    v_measure = metrics.v_measure_score(y_pred, y_true)

    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    r2 = metrics.r2_score(y_true, y_pred)
    logging_info(f"for cls {name} sum, the Adjusted Rand Index (ARI): {ari}")
    logging_info(
        f"for cls {name} sum, the Adjusted Mutual Information (MI): {mi}")
    logging_info(f"for cls {name} sum, the Homogeneity: {homogeneity}")
    logging_info(f"for cls {name} sum, the Completeness: {completeness}")
    logging_info(f"for cls {name} sum, the V-measure: {v_measure}")
    logging_info(f"for cls {name} sum, the rmse: {rmse}")
    logging_info(f"for cls {name} sum, the r2: {r2}")
    global SUM_A, SUM_M, SUM_H, SUM_C, SUM_V, SUM_RMSE, SUM_R2
    SUM_A += ari/NUM_SHAPE_CLUSTER
    SUM_M += mi/NUM_SHAPE_CLUSTER
    SUM_H += homogeneity/NUM_SHAPE_CLUSTER
    SUM_C += completeness/NUM_SHAPE_CLUSTER
    SUM_V += v_measure/NUM_SHAPE_CLUSTER
    SUM_RMSE += rmse/NUM_SHAPE_CLUSTER
    SUM_R2 += r2/NUM_SHAPE_CLUSTER

    logging_info(f"for cls final, the Adjusted Rand Index (ARI): {SUM_A}")
    logging_info(
        f"for cls final, the Adjusted Mutual Information (MI): {SUM_M}")
    logging_info(f"for cls final, the Homogeneity: {SUM_H}")
    logging_info(f"for cls final, the Completeness: {SUM_C}")
    logging_info(f"for cls final, the V-measure: {SUM_V}")
    logging_info(f"for cls final, the SUM_RMSE: {SUM_RMSE}")
    logging_info(f"for cls final, the SUM_R2: {SUM_R2}")


def work_after_cluster(div_results):

    print(f'now work after cluster and calculate texture')

    for ind, left_list in enumerate(div_results):
        lda_labels, features = [], []
        for t_feature, label in left_list:
            lda_labels.append(label)
            features.append(t_feature)

        features = np.array(features, dtype='float32')
        lda_labels = np.array(lda_labels, dtype='uint8')
        n_components = min(len(np.unique(lda_labels))-1, 3)
        print(f'n_components is {n_components}')
        if n_components not in [2, 3]:
            continue
        lda = PCA(n_components=n_components)
        
        if TEST_SIZE != -1:
            x_train, x_test, y_train, y_test = train_test_split(
                features, lda_labels, test_size=TEST_SIZE)
        else:
            x_train, y_train = features, lda_labels
            
        x_train = lda.fit_transform(x_train)
        print(f'finish work for LDA and now start kmeans')
        if n_components == 3:
            x, y, z = x_train[:, 0].tolist(
            ), x_train[:, 1].tolist(), x_train[:, 2].tolist()
        else:
            x, y = x_train[:, 0].tolist(
            ), x_train[:, 1].tolist()

        kmeans = KMeans(n_clusters=NUM_CLUSTER, max_iter=100, init='k-means++')
        ans_cluster = kmeans.fit_predict(x_train)

        with open(os.path.join(TRAIN_DIR, f'{ind}_lda_model.pkl'), 'wb') as f:
            pickle.dump(lda, f)
        with open(os.path.join(TRAIN_DIR, f'{ind}_kmeans_model.pkl'), 'wb') as f:
            pickle.dump(kmeans, f)

        colors = ['red', 'green', 'blue', 'yellow', 'purple',
                  'cyan', 'black', 'gray', 'pink', 'lime']
        fig = plt.figure()
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        ans_num = np.zeros((NUM_CLUSTER, NUM_CLASSES)).astype('int32')

        #  draw for train dataset and test for test dataset
        for i in range(len(x)):
            label = y_train[i]
            if n_components == 3:
                ax.scatter(x[i], y[i], z[i], color=colors[label])
            else:
                ax.scatter(x[i], y[i], color=colors[label])

            ans_num[ans_cluster[i]][label] += 1
        # plt.legend()
        if TEST_SIZE != -1:
            x_test = lda.transform(x_test)
            ans_cluster_test = kmeans.predict(x_test)
        else:
            y_test = y_train
            ans_cluster_test = ans_cluster
            
        check_metric(y_test, ans_cluster_test, str(ind), ans_num)
        plt.savefig(os.path.join(TRAIN_DIR, f'scatter_plot_{ind}.png'))
        print(f'for old cls {ind}, the final result of new cls is \n{ans_num}')
        np.savetxt(os.path.join(
            TRAIN_DIR, f"my_array_{ind}.txt"), ans_num, fmt="%d")


def cluster():

    SOURCE_FEATURE_PATH = './small_features/cluster.npz'
    assert os.path.exists(SOURCE_FEATURE_PATH), 'Error, features not find !'

    data = np.load(SOURCE_FEATURE_PATH)
    features, lda_labels = data['features'], data['lda_labels']
    texture_features = features[:, :-28]
    lda_labels = np.array(lda_labels)
    n_components = min(len(np.unique(lda_labels))-1, 3)
    lda = PCA(n_components=n_components)
    # only last 28 is shape feature
    features = lda.fit_transform(features[:, -28:])
    if n_components == 3:
        x, y, z = features[:, 0].tolist(
        ), features[:, 1].tolist(), features[:, 2].tolist()
    else:
        x, y = features[:, 0].tolist(
        ), features[:, 1].tolist()
    print(f'finish work for lda and now start kmeans')

    kmeans = KMeans(n_clusters=NUM_SHAPE_CLUSTER,
                    max_iter=100, init='k-means++')
    ans_cluster = kmeans.fit_predict(features)
    with open(os.path.join(TRAIN_DIR, 'all_lda_model.pkl'), 'wb') as f:
        pickle.dump(lda, f)
    with open(os.path.join(TRAIN_DIR, 'all_kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)

    colors = ['red', 'green', 'blue', 'yellow', 'purple',
              'cyan', 'black', 'gray', 'pink', 'lime']
    categories = ['A', 'B', 'C', 'D', 'E']
    fig = plt.figure()
    ans_num = np.zeros((NUM_SHAPE_CLUSTER, NUM_CLASSES)).astype('int32')
    show_3d_labels = []
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    for i in range(len(x)):
        label = lda_labels[i]
        show_3d_labels.append(label)
        if n_components == 3:
            ax.scatter(x[i], y[i], z[i], color=colors[label])
        else:
            ax.scatter(x[i], y[i], color=colors[label])
        ans_num[ans_cluster[i]][label] += 1
    # plt.legend()

    plt.savefig(os.path.join(TRAIN_DIR, f'scatter_plot_all.png'))
    np.savetxt(os.path.join(TRAIN_DIR, f"my_array_all.txt"), ans_num, fmt="%d")
    print(ans_num.astype('int32'))
    # np.savez(os.path.join(TRAIN_DIR, f'./show_3D_{NO_USE_NAME}.npz'), x=x, y=y, z=z, ans_cluster=ans_cluster,
    #          labels=np.array(show_3d_labels, dtype='uint8'), ans_num=ans_num)

    div_results = []
    for i in range(NUM_SHAPE_CLUSTER):
        div_results.append([])

    for j, t_feature in enumerate(texture_features):
        div_results[ans_cluster[j]].append([t_feature, lda_labels[j]])

    del features
    del lda_labels

    div_results = sorted(div_results, key=len)
    work_after_cluster(div_results)


def name2number(x):
    new_file = ""
    for c in x:
        if c <= '9' and c >= '0':
            new_file += c
        else:
            break
    return new_file


if __name__ == '__main__':

    os.makedirs(TRAIN_DIR, exist_ok=True)
    with open(LOGGER_PATH, "w") as file:
        file.write("")
    shutil.copy(params, TRAIN_DIR)

    cluster()
    # test_plt()
    # deal_flip()

# -*- coding:utf-8 _*-
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
NUM_PROCESSOR = 96
NUM_CLASSES = 8
NUM_SHAPE_CLUSTER = 8
NUM_CLUSTER = 8
TEST_SIZE = 0.15
TRAIN_DIR = './weights/cluster_other_test'
os.makedirs(TRAIN_DIR, exist_ok=True)

IS_USE = {
    'x': True,  # 圆心的坐标
    'y': True,
    'x1': True,  # 矩形的左上角和右下角
    'y1': True,
    'x2': True,
    'y2': True,
    'extent': True,
    'equi_diameter': True,
    'e_center_x': True,  # 椭圆的坐标
    'e_center_y': True,
}

LOGGER_PATH = os.path.join(TRAIN_DIR, 'train_cluster.txt')
with open(LOGGER_PATH, "w") as file:
    file.write("")


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
img_lbl1 = np.zeros((1024, 256, 3))
for i in range(1024):
    for j in range(256):
        if i <= 100:
            img_lbl1[i, j] = STATIC_COLORS[0]
        if i > 100 and i <= 200:
            img_lbl1[i, j] = STATIC_COLORS[1]
        if i > 200 and i <= 300:
            img_lbl1[i, j] = STATIC_COLORS[2]
        if i > 300 and i <= 400:
            img_lbl1[i, j] = STATIC_COLORS[3]
        if i > 400 and i <= 500:
            img_lbl1[i, j] = STATIC_COLORS[4]
        if i > 500 and i <= 600:
            img_lbl1[i, j] = STATIC_COLORS[5]

cv2.imwrite('show2.jpg', img_lbl1)


@jit(nopython=True)
def work_region_image(img, region):
    res = np.zeros((1024, 256))
    mask = np.zeros((1024, 256))
    for i, x in enumerate(region[0]):
        res[x, region[1][i]] = img[x, region[1][i]]
        mask[x, region[1][i]] = 255
    return res, mask


# prepocess after cluster

params = './CT3D.yaml'
shutil.copy(params, TRAIN_DIR)
all_extractor = featureextractor.RadiomicsFeatureExtractor(params)

SUM_A = 0.0
SUM_M = 0.0
SUM_H = 0.0
SUM_C = 0.0
SUM_V = 0.0


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
    logging_info(f"for cls {name} sum, the Adjusted Rand Index (ARI): {ari}")
    logging_info(
        f"for cls {name} sum, the Adjusted Mutual Information (MI): {mi}")
    logging_info(f"for cls {name} sum, the Homogeneity: {homogeneity}")
    logging_info(f"for cls {name} sum, the Completeness: {completeness}")
    logging_info(f"for cls {name} sum, the V-measure: {v_measure}")
    global SUM_A, SUM_M, SUM_H, SUM_C, SUM_V
    SUM_A += ari/NUM_SHAPE_CLUSTER
    SUM_M += mi/NUM_SHAPE_CLUSTER
    SUM_H += homogeneity/NUM_SHAPE_CLUSTER
    SUM_C += completeness/NUM_SHAPE_CLUSTER
    SUM_V += v_measure/NUM_SHAPE_CLUSTER
    logging_info(f"for cls final, the Adjusted Rand Index (ARI): {SUM_A}")
    logging_info(
        f"for cls final, the Adjusted Mutual Information (MI): {SUM_M}")
    logging_info(f"for cls final, the Homogeneity: {SUM_H}")
    logging_info(f"for cls final, the Completeness: {SUM_C}")
    logging_info(f"for cls final, the V-measure: {SUM_V}")


def find_dict(cur, ans):
    for k, v in cur.items():
        if isinstance(v, dict):
            find_dict(v, ans)
        else:
            ans[k] = v


def prepocess_after(img, mask, isTest=False):
    if not isTest:
        mask = (mask/255.0).astype('uint8')

    data_spacing = [1, 1, 1]

    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetSpacing((float(data_spacing[0]), float(
        data_spacing[1]), float(data_spacing[2])))
    sitk_img = sitk.JoinSeries(sitk_img)
    sitk_mask = sitk.GetImageFromArray(mask)
    sitk_mask.SetSpacing((float(data_spacing[0]), float(
        data_spacing[1]), float(data_spacing[2])))
    sitk_mask = sitk.JoinSeries(sitk_mask)
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32)
    features = all_extractor.execute(
        sitk_img, sitk_mask, label=1, voxelBased=True)

    res_features = {}
    res = []
    find_dict(features, res_features)
    del features
    for k, v in res_features.items():
        if isinstance(v, tuple):
            res.extend(list(v))
        if isinstance(v, int):
            res.append(v)
        if isinstance(v, list):
            res.extend(v)
        if isinstance(v, sitk.SimpleITK.Image):
            img_array = sitk.GetArrayFromImage(v)
            try:
                U, s, Vt = np.linalg.svd(img_array)
                s = np.sum(s**2)
                eigenvalue = np.sqrt(s)
            except:
                eigenvalue = 0
            res.append(eigenvalue)
    print(len(res))
    del res_features

    return np.array(res, dtype=np.float32)


def test_prepocess_after():
    img = np.random.rand(1024, 256)
    mask = np.zeros((1024, 256))
    for i in range(300, 400, 1):
        for j in range(100, 200, 1):
            mask[i, j] = 1
    res = prepocess_after(img, mask, True)
    print(res.shape)

from multiprocessing import Pool
from typing import List
import gc
from time import sleep

def check_workers_alive_and_busy(export_pool: Pool, worker_list: List, results_list: List, allowed_num_queued: int = 0):
    """

    returns True if the number of results that are not ready is greater than the number of available workers + allowed_num_queued
    """
    alive = [i.is_alive() for i in worker_list]
    if not all(alive):
        raise RuntimeError('Some background workers are no longer alive')

    not_ready = [not i.ready() for i in results_list]

    if sum(not_ready) >= (len(export_pool._pool) + allowed_num_queued):
        return True
    return False


def work_after_cluster(div_results):

    print(f'now work after cluster and calculate texture')

    for ind, cls in enumerate(div_results):
        results = []
        lda_labels = []
        with mp.get_context('spawn').Pool(processes=NUM_PROCESSOR) as pool:
            worker_list = [i for i in pool._pool]
            for img, mask, label, name in tqdm(cls):
                proceed = not check_workers_alive_and_busy(
                    pool, worker_list, results, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        pool, worker_list, results, allowed_num_queued=2)
                gc.collect()
                lda_labels.append(label)
                result = pool.starmap_async(
                    prepocess_after, ((img, mask),))
                results.append(result)

            features = [r.get()[0] for r in results]
        features = np.array(features)
        print(
            f'finish prepocess data for cls {ind}, length is {features.shape}')

        lda_labels = np.array(lda_labels)
        n_components = min(len(np.unique(lda_labels))-1, 3)
        print(f'n_components is {n_components}')
        print(lda_labels.shape)
        if n_components not in [2, 3]:
            continue

        lda = LDA(n_components=n_components)
        features = lda.fit_transform(features, lda_labels)
        print(f'finish work for PCA and now start kmeans')
        if n_components == 3:
            x, y, z = features[:, 0].tolist(
            ), features[:, 1].tolist(), features[:, 2].tolist()
        else:
            x, y = features[:, 0].tolist(
            ), features[:, 1].tolist()
        # features = torch.tensor(features).cuda()
        # kmeans = KMeans(n_clusters=5, max_iter=100,
            # mode='euclidean', verbose=1)
        # ans_cluster = kmeans.fit_predict(features).detach().cpu().numpy()

        kmeans = KMeans(n_clusters=NUM_CLUSTER, max_iter=100, init='k-means++')
        ans_cluster = kmeans.fit_predict(features)

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
            label = lda_labels[i]
            if n_components == 3:
                ax.scatter(x[i], y[i], z[i], color=colors[label],)
            else:
                ax.scatter(x[i], y[i], color=colors[label])

            ans_num[ans_cluster[i]][label] += 1
        # plt.legend()
        check_metric(lda_labels, ans_cluster, str(ind), ans_num)
        plt.savefig(os.path.join(TRAIN_DIR, f'scatter_plot_{ind}.png'))
        print(f'for old cls {ind}, the final result of new cls is \n{ans_num}')
        np.savetxt(os.path.join(
            TRAIN_DIR, f"my_array_{ind}.txt"), ans_num, fmt="%d")


def prepocess(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    gray = img.copy().astype('uint8')
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 得到多个轮廓，取最大的那个
    contours_ans = contours[0]
    if len(contours) != 1:
        idx, mx = 0, 0
        for i in range(len(contours)):
            if mx < len(contours[i]):
                mx = len(contours[i])
                idx = i
    
        contours_ans = contours[idx]
   # sys.exit(1)

    # 计算曲线近似椭圆
    try:
        ellipse = cv2.fitEllipse(contours_ans)
        e_center_x, e_center_y, e_max_axis, e_min_axis, e_angle = ellipse[0][0], ellipse[0][1], max(
            ellipse[1]), min(ellipse[1]), ellipse[2]
        curvature = 1 / ellipse[1][0]
        eccentricity = np.sqrt(1 - (e_min_axis**2 / e_max_axis**2))

    except:
        e_center_x, e_center_y, e_max_axis, e_min_axis, e_angle = 0.0, 0.0, 0.0, 0.0, 0.0
        curvature = 0
        eccentricity = 1

    # 计算曲线弧度
    [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
    slope = vy / vx
    # angle = np.arctan(slope)
    # 计算面积
    area = cv2.contourArea(contours_ans)
    # 计算周长
    # perimeter = cv2.arcLength(contours_ans, True)
    cnt = contours_ans
    # 外围矩形信息
    x, y, w, h = cv2.boundingRect(cnt)
    x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
    # 长宽比
    # aspect_ratio = float(w)/h
    # 矩形面积
    rect_area = w*h
    # 矩形面积比
    extent = float(area)/rect_area
    # 等效直径
    equi_diameter = np.sqrt(4*area/np.pi)
    # print(f'area:{area}\n perimeter:{perimeter}\n aspect_ratio:{aspect_ratio}\n extent:{extent}\n equi_diameter:{equi_diameter}\n')
    (x, y), r = cv2.minEnclosingCircle(contours_ans)
    # 计算等圆性
    # circular_area = (np.pi * r**2)
    # circularity = area / (np.pi * r**2)
    # 计算等周性
    # circular_perimeter = (np.pi * r * 2)
    # equidiameter = perimeter/(np.pi * r * 2)

    res = []
    for k, v in IS_USE.items():
        if v == True:
            if np.isnan(eval(k)).any():
                res.append(0)
            else:
                res.append(eval(k))

    # res = [x, y, w, h, area, perimeter, aspect_ratio,
    #        rect_area, extent, equi_diameter, e_center_x, e_center_y, e_max_axis, e_min_axis, e_angle, angle, curvature, eccentricity, circular_area, circularity, circular_perimeter, equidiameter]
    # print(res)
    # res += img.astype('float32').reshape(-1).tolist()
    # print(res)
    return np.array(res, dtype=np.float32)


def prepocess_all(img, mask):
    a = prepocess(np.copy(mask))
    # b = prepocess_after(img, mask)
    # return np.concatenate((a, b))
    return a


@jit(nopython=True)
def label_work_run(lbl, label_array):
    lbl_res = np.zeros((1024, 256))
    img_lbl_res = np.zeros((1024, 256, 3))
    for i in range(1024):
        for j in range(256):
            x = lbl[i, j]
            lbl_res[i, j] = label_array[x]
    return lbl_res, img_lbl_res


def label_work(lbl, LABEL_TO_INDEX, revers_dict):

    label_array = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    for k, v in revers_dict.items():
        label_array[k] = LABEL_TO_INDEX[v]

    return label_work_run(lbl, label_array)


def get_shape_features(img: np.ndarray, lbl: np.ndarray, file_name: str):
    all_image = []
    left_info = []
    features = []
    lda_labels = []
    mask = np.zeros_like(lbl)
    for label in range(1, NUM_CLASSES+1):
        # only extract equal pixel
        lbl_b = (lbl == label)*255
        ret, thresh = cv2.threshold(lbl_b.astype(
            'uint8'), 127, 255, cv2.THRESH_BINARY)
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_mask = np.zeros_like(lbl)
            cv2.drawContours(contour_mask, [contour], 0, 1, -1)
            mask += contour_mask
            num_t = np.count_nonzero((contour_mask.astype('uint8') == 1)*1)
            if num_t <= 10:
                # print('skip this noise')
                continue
            # cv2.imwrite('./test_show.jpg', (contour_mask*255).astype('uint8'))
            lda_labels.append(label-1)
            features.append(prepocess(contour_mask*255))
            all_image.append(contour_mask*img)
            left_info.append([label-1, file_name, contour_mask*255])
    # ans = np.concatenate(
        # [(mask*255).astype('uint8'), ((lbl != 0)*255).astype('uint8')], axis=1)
    # cv2.imwrite('./test_mask.jpg', ans)
    return lda_labels, all_image, left_info, features


def cluster(NO_USE_NAME):

    SOURCE_DIR_PATH = './other_data'

    all_image = []
    left_info = []
    features = []
    lda_labels = []
    for path, dir_list, file_list in os.walk(SOURCE_DIR_PATH):
        for file_name in tqdm(file_list):
            if file_name.find('.npz') == -1:
                continue

            data = np.load(os.path.join(path, file_name))
            img, lbl = data['image'], data['label_a']
            if len(np.unique(lbl)) == 0:
                continue
            lda_labels_item, all_image_item, left_info_item, feature_item = get_shape_features(
                img, lbl, file_name)
            lda_labels.extend(lda_labels_item)
            all_image.extend(all_image_item)
            left_info.extend(left_info_item)
            features.extend(feature_item)

    features = np.array(features)
    np.savez(os.path.join(TRAIN_DIR, './first_features.npz'), data=features)
    print(f'finish load data, length is {features.shape}')
    lda_labels = np.array(lda_labels)
    n_components = min(len(np.unique(lda_labels))-1, 3)
    lda = LDA(n_components=n_components)
    features = lda.fit_transform(features, lda_labels)
    if n_components == 3:
        x, y, z = features[:, 0].tolist(
        ), features[:, 1].tolist(), features[:, 2].tolist()
    else:
        x, y = features[:, 0].tolist(
        ), features[:, 1].tolist()
    print(f'finish work for PCA and now start kmeans')

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
        label, name = left_info[i][0], left_info[i][1]
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

    for j, img in enumerate(all_image):
        label, name, region_mask, cluster = left_info[j][0], left_info[j][1], left_info[j][2], ans_cluster[j]
        div_results[cluster].append([img, region_mask, label, name])

    del all_image
    del left_info
    del features
    del lda_labels

    div_results = sorted(div_results, key=len)
    work_after_cluster(div_results)


def cluster_all():
    NO_USE_NAME = ''
    cluster(NO_USE_NAME)
    # for k, v in IS_USE.items():
    #     tmp_res = deepcopy(IS_USE)
    #     tmp_res[k] = False
    #     cluster(tmp_res, k)
    #     print(f'finish work cluster for no use {k}')


def name2number(x):
    new_file = ""
    for c in x:
        if c <= '9' and c >= '0':
            new_file += c
        else:
            break
    return new_file


if __name__ == '__main__':

    # deep2_filp()
    # test_prepocess_after()
    cluster_all()
    # test_plt()
    # deal_flip()

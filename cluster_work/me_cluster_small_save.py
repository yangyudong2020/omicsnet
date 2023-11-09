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
NUM_PROCESSOR = 96
NUM_CLASSES = 4
NUM_SHAPE_CLUSTER = 4
NUM_CLUSTER = 4
TEST_SIZE = 0.15
TRAIN_DIR = './weights/me_small_train_1'
LOGGER_PATH = f'{TRAIN_DIR}/train_cluster.txt'
FULL_WIDTH = 384
FULL_HEIGHT = 768
params = './CT3D.yaml'
all_extractor = featureextractor.RadiomicsFeatureExtractor(params)


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


def find_dict(cur, ans, pre_key=''):
    for k, v in cur.items():
        if isinstance(v, dict):
            find_dict(v, ans, pre_key+'_'+k)
        else:
            ans[pre_key+'_'+k] = v


def prepocess_after(img, mask, isTest=False):
    if not isTest:
        mask = (mask/255.0).astype('uint8')

    rows, cols = np.where(mask == 255)

    # 计算坐标的均值
    x = np.mean(rows)
    y = np.mean(cols)

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
    res_shape = [y, x]
    res_texture = []
    res_value = []
    find_dict(features, res_features)
    del features
    # print('len feature is ', len(res_features))
    for k, v in res_features.items():
        # print('key is', k, 'type is', type(v))
        if k.find('Versions') != -1 or k.find('Configuration') != -1 or k.find('interpolated') != -1:
            continue

        if k.find('Image') != -1 or k.find('Mask') != -1:
            if isinstance(v, tuple):
                res_shape.extend(list(v))
            if isinstance(v, int):
                res_shape.append(v)
            if isinstance(v, list):
                res_shape.extend(v)
        else:
            img_array = sitk.GetArrayFromImage(v)
            res_texture.append(img_array)
            try:
                U, s, Vt = np.linalg.svd(img_array)
                u, s, vt = np.sum(s**2), np.sum(U**2), np.sum(Vt**2)
                eigenvalue = [np.sqrt(s), np.sqrt(u), np.sqrt(vt)]
            except:
                print('error value')
                eigenvalue = [0, 0, 0]
            res_value.extend(eigenvalue)
    # print(len(res))
    del res_features
    res_value.extend(res_shape)
    res_value = np.array(res_value, dtype=np.float32)
    res_texture, res_shape = np.array(
        res_texture, dtype=np.float32), np.array(res_shape, dtype=np.float32)
    res_texture = res_texture.squeeze()
    return [res_texture, res_shape, res_value]


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

    results = []
    lda_labels = []
    # div_results = div_results[:2]
    with mp.get_context('spawn').Pool(processes=NUM_PROCESSOR) as pool:
        worker_list = [i for i in pool._pool]
        for img, mask, label, name in tqdm(div_results):
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

    # res_texture, res_shape = features[i]
    item_features = {}
    max_len = 0
    res_value_list = []
    for i, (img, mask, label, name) in enumerate(div_results):
        res_texture, res_shape, res_value = features[i]
        res_value_list.append(res_value)

    res_value_list = np.array(res_value_list, dtype='float32')
    lda_labels = np.array(lda_labels, dtype='uint8')
    res_value_list[np.isnan(res_value_list)] = 0
    np.savez('small_features/cluster.npz',
            features=res_value_list, lda_labels=lda_labels)
    # using for normalize
    
    # 沿第二维计算最大值和最小值
    min_vals = np.min(res_value_list, axis=1, keepdims=True)
    max_vals = np.max(res_value_list, axis=1, keepdims=True)

    # 归一化到 [0,1] 范围
    normalized_arr = (res_value_list - min_vals) / (max_vals - min_vals)

    # 调整到 [-1,1] 范围
    res_value_list = 2 * normalized_arr - 1

    for i, (img, mask, label, name) in enumerate(div_results):
        if name not in item_features:
            item_features[name] = {
                'texture': [],
                'shape': [],
                'value': []
            }
        res_value = res_value_list[i]
        # item_features[name]['texture'].append(res_texture)
        # item_features[name]['shape'].append(res_shape)
        item_features[name]['value'].append(res_value)
    for k in item_features:
        # item_features[k]['texture'] = np.array(item_features[k]['texture'])
        # item_features[k]['shape'] = np.array(item_features[k]['shape'])
        item_features[k]['value'] = np.array(item_features[k]['value'])
        # new_name = k[:-4]+'_radio.npz'
        # np.savez('small_features/'+new_name,
        #          shape=item_features[k]['shape'],
        #          value=item_features[k]['value'])
        # np.save('small_features/'+k[:-4] +
        #         '_radio.npy', item_features[k]['value'])


def get_contour_position(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return y, x


def get_item_image(img: np.ndarray, lbl: np.ndarray, file_name: str):
    all_image = []
    left_info = []
    mask = np.zeros_like(lbl)
    for label in range(1, NUM_CLASSES+1):
        # only extract equal pixel
        lbl_b = (lbl == label)*255
        ret, thresh = cv2.threshold(lbl_b.astype(
            'uint8'), 127, 255, cv2.THRESH_BINARY)
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=get_contour_position)

        for contour in contours:
            contour_mask = np.zeros_like(lbl)
            cv2.drawContours(contour_mask, [contour], 0, 1, -1)
            mask += contour_mask
            num_t = np.count_nonzero((contour_mask.astype('uint8') == 1)*1)
            if num_t <= 10:
                # print('skip this noise')
                continue
            # cv2.imwrite('./test_show.jpg', (contour_mask*255).astype('uint8'))
            all_image.append(contour_mask*img)
            left_info.append([label-1, file_name, contour_mask*255])

    return all_image, left_info


def cluster():

    SOURCE_DIR_PATH = './real_my_data_small'

    all_image = []
    left_info = []
    for path, dir_list, file_list in os.walk(SOURCE_DIR_PATH):
        for file_name in tqdm(file_list):
            if file_name.find('.npz') == -1:
                continue

            data = np.load(os.path.join(path, file_name))
            img, lbl = data['image'].astype('int32'), data['label_a']
            if len(np.unique(lbl)) == 0:
                continue
            all_image_item, left_info_item = get_item_image(
                img, lbl, file_name)
            all_image.extend(all_image_item)
            left_info.extend(left_info_item)

    div_results = []
    for j, img in enumerate(all_image):
        label, name, region_mask = left_info[j][0], left_info[j][1], left_info[j][2]
        div_results.append([img, region_mask, label, name])

    del all_image
    del left_info

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

    # deep2_filp()
    # test_prepocess_after()
    os.system('rm -rf ./small_features/*')
    os.makedirs('small_features', exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    with open(LOGGER_PATH, "w") as file:
        file.write("")
    shutil.copy(params, TRAIN_DIR)

    cluster()
    # test_plt()
    # deal_flip()

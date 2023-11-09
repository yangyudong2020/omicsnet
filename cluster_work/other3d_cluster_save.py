# -*- coding:utf-8 _*-
# from now, we only deal data for nnUNet_raw, and feature worker will
# generate featuresTr dir
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
import json
import cv2
import numpy as np
import os
import SimpleITK as sitk
import os
from numba import jit

# import
NUM_PROCESSOR = 3
NUM_CLASSES = 2
NUM_SHAPE_CLUSTER = 2
NUM_CLUSTER = 2
SOURCE_DIR_PATH = 'Dataset005_OtherPet'
TRAIN_DIR = './weights/otherPet'
LOGGER_PATH = f'{TRAIN_DIR}/train_cluster.txt'
FULL_WIDTH = 100
FULL_HEIGHT = 100
params = './CT3D.yaml'
all_extractor = featureextractor.RadiomicsFeatureExtractor(params)


def logging_info(x):
    with open(LOGGER_PATH, "a") as file:
        file.write(x+'\n')
    print(x)


def find_dict(cur, ans, pre_key=''):
    for k, v in cur.items():
        if isinstance(v, dict):
            find_dict(v, ans, pre_key+'_'+k)
        else:
            ans[pre_key+'_'+k] = v


def prepocess_after(img, mask):

    rows, cols, zs = np.where(mask == 1)

    # 计算坐标的均值
    x = np.mean(rows)
    y = np.mean(cols)
    z = np.mean(zs)

    data_spacing = [1, 1, 1]

    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetSpacing((float(data_spacing[0]), float(
        data_spacing[1]), float(data_spacing[2])))
    # sitk_img = sitk.JoinSeries(sitk_img)
    sitk_mask = sitk.GetImageFromArray(mask)
    sitk_mask.SetSpacing((float(data_spacing[0]), float(
        data_spacing[1]), float(data_spacing[2])))
    # sitk_mask = sitk.JoinSeries(sitk_mask)
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32)
    sitk_mask_array = sitk.GetArrayFromImage(sitk_mask)
    # print(f'mask unique is {np.unique(sitk_mask_array)}')
    features = all_extractor.execute(
        sitk_img, sitk_mask, label=1, voxelBased=True)

    res_features = {}
    res_shape = [y, x, z]
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
    # print('res_value len is', len(res_value), 'res_shape len is', len(res_shape))
    res_value = np.array(res_value, dtype=np.float32)
    return [None, res_shape, res_value]


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
    cluster_path = os.environ.get(
        'nnUNet_raw')+f'/{SOURCE_DIR_PATH}/cluster.npz'
    np.savez(cluster_path, features=res_value_list, lda_labels=lda_labels)
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
    item_save_path = os.environ.get(
        'nnUNet_raw')+f'/{SOURCE_DIR_PATH}/featuresTr/'
    os.makedirs(item_save_path, exist_ok=True)
    for k in item_features:
        # item_features[k]['texture'] = np.array(item_features[k]['texture'])
        # item_features[k]['shape'] = np.array(item_features[k]['shape'])
        item_features[k]['value'] = np.array(item_features[k]['value'])
        # np.savez('small_features/'+new_name,
        #          shape=item_features[k]['shape'],
        #          value=item_features[k]['value'])

        np.save(item_save_path + k[:-7] +
                '.npy', item_features[k]['value'])


def get_contour_position(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return y, x


def get_item_image(img: np.ndarray, lbl: np.ndarray, file_name: str):
    all_image = []
    left_info = []

    for label in range(1, NUM_CLASSES+1):
        # only extract equal pixel
        lbl_b = (lbl == label)*1

        mask_label = sitk.GetImageFromArray(lbl_b)
        labeled_mask = sitk.ConnectedComponent(mask_label)
        relabel_image = sitk.RelabelComponent(labeled_mask)
        # print('relabel shape is', sitk.GetArrayFromImage(relabel).shape)
        num_blocks = relabel_image.GetNumberOfComponentsPerPixel()
        print(f'number of {num_blocks} for label {label}')
        # 遍历每个块
        for block_id in range(1, num_blocks + 1):
            # 提取当前块
            block = sitk.BinaryThreshold(
                relabel_image, lowerThreshold=block_id, upperThreshold=block_id)
            block_array = sitk.GetArrayFromImage(block)
            if np.sum(block_array > 0) <= 10:
                continue
            # cv2.imwrite('./test_show.jpg', (contour_mask*255).astype('uint8'))
            all_image.append(block_array*img)
            left_info.append([label-1, file_name, block_array])
            # block_array is a mask which is 0 or 1

    return all_image, left_info


def cluster():

    all_image = []
    left_info = []
    image_dir_path = os.environ.get(
        'nnUNet_raw')+f'/{SOURCE_DIR_PATH}/imagesTr'
    label_dir_path = image_dir_path.replace('images', 'labels')
    for path, dir_list, file_list in os.walk(image_dir_path):
        for file_name in tqdm(file_list):
            img = sitk.ReadImage(os.path.join(image_dir_path, file_name))
            img = sitk.GetArrayFromImage(img)
            lbl = sitk.ReadImage(os.path.join(
                label_dir_path, file_name.replace('_0000', '')))
            lbl = sitk.GetArrayFromImage(lbl)

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

    os.makedirs(TRAIN_DIR, exist_ok=True)
    with open(LOGGER_PATH, "w") as file:
        file.write("")
    shutil.copy(params, TRAIN_DIR)

    cluster()
    # test_plt()
    # deal_flip()

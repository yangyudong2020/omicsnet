# -*- coding:utf-8 _*-
import gc
import random
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
# from kmeans_pytorch import kmeans
# from fast_pytorch_kmeans import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
import pywt
import sys
from labelme.utils import *
from PIL import Image, ImageDraw
import h5py
from skimage.draw import line_aa, circle_perimeter_aa
from skimage import draw
import base64
import json
import cv2
from urllib import request
import numpy as np
import os
import SimpleITK as sitk
import os
from numba import jit


@jit(nopython=True)
def work_region_image(img, region):
    res = np.zeros((1024, 256))
    mask = np.zeros((1024, 256))
    for i, x in enumerate(region[0]):
        res[x, region[1][i]] = img[x, region[1][i]]
        mask[x, region[1][i]] = 255
    return res, mask


# prepocess after cluster
DIFF_LIST = []
for diff_x in range(5, 21):
    for diff_y in range(diff_x, 21):
        DIFF_LIST.append([diff_x, diff_y])

params = './CT3D.yaml'
all_extractor = featureextractor.RadiomicsFeatureExtractor(params)
NUM_PROCESSOR = 48


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
    del features
    del res_features
    gc.collect()
    return np.array(res, dtype=np.float32)


def prepocess(img, IS_USE=None):
    IS_USE = {
        'x': True,  # 圆心的坐标
        'y': True,
        'w': True,
        'h': True,
        'x1': True,  # 矩形的左上角和右下角
        'y1': True,
        'x2': True,
        'y2': True,
        'area': False,
        'perimeter': False,
        'aspect_ratio': False,
        'rect_area': False,
        'extent': True,
        'equi_diameter': True,
        'e_center_x': True,  # 椭圆的坐标
        'e_center_y': True,
        'e_max_axis': True,
        'e_min_axis': True,
        'e_angle': True,
        'angle': True,
        'curvature': True,
        'eccentricity': True,
        'circular_area': False,
        'circularity': True,
        'circular_perimeter': False,
        'equidiameter': True,
    }
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    img = img[:, :, np.newaxis]
    img = np.tile(img, (1, 1, 3)).astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    angle = np.arctan(slope)
    # 计算面积
    area = cv2.contourArea(contours_ans)
    # 计算周长
    perimeter = cv2.arcLength(contours_ans, True)
    cnt = contours_ans
    # 外围矩形信息
    x, y, w, h = cv2.boundingRect(cnt)
    x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
    # 长宽比
    aspect_ratio = float(w)/h
    # 矩形面积
    rect_area = w*h
    # 矩形面积比
    extent = float(area)/rect_area
    # 等效直径
    equi_diameter = np.sqrt(4*area/np.pi)
    # print(f'area:{area}\n perimeter:{perimeter}\n aspect_ratio:{aspect_ratio}\n extent:{extent}\n equi_diameter:{equi_diameter}\n')
    (x, y), r = cv2.minEnclosingCircle(contours_ans)
    # 计算等圆性
    circular_area = (np.pi * r**2)
    circularity = area / (np.pi * r**2)
    # 计算等周性
    circular_perimeter = (np.pi * r * 2)
    equidiameter = perimeter/(np.pi * r * 2)

    res = []
    for k, v in IS_USE.items():
        if v == True:
            res.append(eval(k))

    # res = [x, y, w, h, area, perimeter, aspect_ratio,
    #        rect_area, extent, equi_diameter, e_center_x, e_center_y, e_max_axis, e_min_axis, e_angle, angle, curvature, eccentricity, circular_area, circularity, circular_perimeter, equidiameter]
    # print(res)
    # res += img.astype('float32').reshape(-1).tolist()

    return np.array(res, dtype=np.float32)


def name2number(x):
    new_file = ""
    for c in x:
        if c <= '9' and c >= '0':
            new_file += c
        else:
            break
    return new_file


@jit(nopython=True)
def label_work_run(lbl, label_array):
    lbl_res = np.zeros((1024, 256))
    img_lbl_res = np.zeros((1024, 256, 3))
    for i in range(1024):
        for j in range(256):
            x = lbl[i, j]
            lbl_res[i, j] = label_array[x]
    return lbl_res, img_lbl_res


@jit(nopython=True)
def find_queue(img):
    queue = []
    # 对十个点进行扩散
    for x in range(1024):
        for y in range(256):
            if img[x, y] == 255:
                queue.append([x, y])
    return queue


def adjust(img, queue):
    if queue == None:
        queue = find_queue(img)
    cnt = 5
    random.shuffle(queue)
    while True:
        cnt -= 1
        x, y = queue.pop(0)
        if x > 0 and img[x-1, y] != img[x, y]:
            img[x-1, y] = 255
            queue.append([x - 1, y])
        if x < img.shape[0]-1 and img[x+1, y] != img[x, y]:
            img[x+1, y] = 255
            queue.append([x + 1, y])
        if y > 0 and img[x, y-1] != img[x, y]:
            img[x, y-1] = 255
            queue.append([x, y - 1])
        if y < img.shape[1]-1 and img[x, y+1] != img[x, y]:
            img[x, y+1] = 255
            queue.append([x, y + 1])
        if cnt == 0:
            break
    return img, queue


def label_work(lbl, LABEL_TO_INDEX, revers_dict):

    label_array = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    for k, v in revers_dict.items():
        label_array[k] = LABEL_TO_INDEX[v]

    return label_work_run(lbl, label_array)


def get_posible(mask):
    # mask is 0-255
    mask = mask.astype(np.uint8)
    ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_ans = contours[0]
    if len(contours) != 1:
        idx, mx = 0, 0
        for i in range(len(contours)):
            if mx < len(contours[i]):
                mx = len(contours[i])
                idx = i

        contours_ans = contours[idx]
    # 创建轮廓掩膜图像
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contours_ans], 0, 255, -1)
    posible_list = [contour_mask]
    # posible_list.extend()
    # queue = None
    # for i in range(100):
    #     contour_mask, queue = adjust(contour_mask, queue)
    #     posible_list.append(np.copy(contour_mask))
    return posible_list


def get_features(lda_list, kmeans_list, region_img, region_mask):
    features = prepocess(np.copy(region_mask)).reshape(1, -1)
    features1 = lda_list[0].transform(features)
    ans1 = kmeans_list[0].predict(features1)[0]
    features = prepocess_after(region_img, region_mask).reshape(1, -1)
    features2 = lda_list[ans1+1].transform(features)
    ans2 = kmeans_list[ans1+1].predict(features2)[0]
    return features1, features2, ans1, ans2


def get_query(lda_list, kmeans_list, img1, img2, lbl2, x, y, label, region_img, region_mask, diff_l, diff_u):
    ans = [False, -1, -1, -1, -1, -1]
    try:
        seed = [y, x]
        img_t = (img2*(region_mask/255.0)).astype(np.uint8)
        output = np.zeros((1026, 258)).astype(np.uint8)
        cv2.floodFill(img_t, output, seed, 1, diff_l, diff_u)
        output.dtype = np.bool_
        output = (~output[1:1025, 1:257])*255
        output = get_posible(output)[0]/255
        # spread image
        # cv2.imwrite('./test_show_origin.jpg', img_t)
        # cv2.imwrite('./test_show_spread.jpg', img_t)
        # cv2.imwrite('./test_show_regionmask.jpg', region_mask)
        # cv2.imwrite('./test_show_output.jpg', output*255)
        # sys.exit(1)
        f2x, f2y, a2x, a2y = get_features(
            lda_list, kmeans_list, output*img_t, output*255)
        ans = [True, f2x, f2y, a2x, a2y, [
               (img1*(region_mask/255.0)).astype(np.uint8), img_t, output*img_t, output*255]]
    except:
        pass
    return ans


def work_deep(lda_list, kmeans_list, img1, points_list, img2, lbl2):
    all_flip = []
    pool = mp.Pool(processes=NUM_PROCESSOR)
    results = []
    for ind, item in tqdm(enumerate(points_list), total=len(points_list)):
        x, y, label, region_img, region_mask = item
        if not (lbl2[x, y] != label and lbl2[x, y] == 0):
            continue

        f1x, f1y, a1x, a1y = get_features(
            lda_list, kmeans_list, region_img, region_mask)
        
        results.append([[f1x, f1y, a1x, a1y]])
        # 多次实验随机种子扩散
        for diff_l, diff_u in DIFF_LIST:
            # ans = get_query(lda_list, kmeans_list, img1, img2,
            # lbl2, x, y, label, region_img, region_mask, diff_l, diff_u)

            ans = pool.apply_async(get_query, args=(lda_list, kmeans_list, img1, img2,
                                                    lbl2, x, y, label, region_img, region_mask, diff_l, diff_u))
            results[len(results) - 1].append(ans)

    pool.close()
    pool.join()

    for result in results:
        f1x, f1y, a1x, a1y = result[0]
        get_list = result[1:]
        ans_list = []
        mx_dist = -1
        for r in get_list:
            is_use, f2x, f2y, a2x, a2y, imgs = r.get()
            if is_use == False:
                continue

            if a2x == a1x and a2y == a1y:
                cur_dist = np.linalg.norm(f1x - f2x)+np.linalg.norm(f1y - f2y)
                if mx_dist == -1 or cur_dist < mx_dist:
                    mx_dist = cur_dist
                    ans_list = imgs

        if len(ans_list) != 0:
            all_flip.append([deepcopy(ans_list), True])

        else:
            all_flip.append([deepcopy(ans_list), False])

        del ans_list

    return all_flip


def deep2_filp():

    SOURCE_DIR_PATH = './data_from'
    TARGET_DIR_PATH = './infer_result_origin'
    TRAIN_DIR = './shape_texture_1'
    USE_DCM = True
    LABEL_TO_INDEX = {
        '_background_': 0,
        'M': 1,
        'N': 2,
        'U': 3,
        'L': 4,
        'F': 5
    }
    os.makedirs(TARGET_DIR_PATH, exist_ok=True)
    with open(os.path.join(TARGET_DIR_PATH, './logger.txt'), 'w') as f:
        f.write('\n')
    lda_list, kmeans_list = [], []
    open_list = ['all', '0', '1', '2', '3', '4']
    for x in open_list:
        with open(os.path.join(TRAIN_DIR, f'./{x}_lda_model.pkl'), 'rb') as f:
            lda = pickle.load(f)
        with open(os.path.join(TRAIN_DIR, f'./{x}_kmeans_model.pkl'), 'rb') as f:
            kmeans = pickle.load(f)
        lda_list.append(lda)
        kmeans_list.append(kmeans)

    t_num, f_num = 0, 0
    for path, dir_list, file_list in os.walk(SOURCE_DIR_PATH):
        for file_name in file_list:
            if file_name.find('.json') == -1:
                continue
            with open(os.path.join(SOURCE_DIR_PATH, file_name), 'r', encoding='utf-8') as f:
                s = f.read()
                all_data = json.loads(s)
            new_name = name2number(file_name)
            data1, data2 = all_data['data1'], all_data['data2']
            if not USE_DCM:
                img1 = img_b64_to_arr(data1['imageData'])[:, :, 0:4]
                img1 = cv2.resize(img1, (256, 1024))[:, :, 0]
                img2 = img_b64_to_arr(data2['imageData'])[:, :, 0:4]
                img2 = cv2.resize(img2, (256, 1024))[:, :, 0]
            else:
                dcm_data = sitk.ReadImage(os.path.join(path, new_name+'.dcm'))
                spacing = dcm_data.GetSpacing()
                dcm_data = sitk.GetArrayFromImage(dcm_data)
                img1, img2 = dcm_data[0], dcm_data[1]

            shapes1 = data1['shapes']
            points_list = []
            for i, shape in enumerate(shapes1):
                label = shape['label']
                label = LABEL_TO_INDEX[label]
                points = shape['points']
                mask = shape_to_mask(
                    img1.shape, points, shape_type=shape.get('shape_type', None))
                region = np.where(mask)
                region_img, region_mask = work_region_image(img1, region)

                max_index = np.argmax(region_img)
                x, y = np.unravel_index(max_index, region_img.shape)
                points_list.append([x, y, label, region_img, region_mask])

            lbl2, lbl_names2 = labelme_shapes_to_label(
                img2.shape, data2['shapes'])
            revers_dict = {}
            for k, v in lbl_names2.items():
                revers_dict[v] = k
            lbl2, img_lbl2 = label_work(
                lbl2, LABEL_TO_INDEX, revers_dict)

            lbl2 = cv2.flip(lbl2, 1)
            img2 = cv2.flip(img2, 1)
            all_flip = work_deep(lda_list, kmeans_list,
                                 img1, points_list, img2, lbl2)

            file_dir_path = os.path.join(TARGET_DIR_PATH, file_name)
            os.makedirs(file_dir_path, exist_ok=True)
            for i, ans_item in enumerate(all_flip):
                ans_list, is_true = ans_item
                label, region_mask = points_list[i][2], points_list[i][4]
                if not is_true:
                    f_num += 1
                    if not USE_DCM:
                        img_path = os.path.join(
                            file_dir_path, f'{i}_no_{label}.jpg')
                        img_res = np.concatenate(
                            [(region_mask/255)*img1, (region_mask/255)*img2], axis=1)
                        cv2.imwrite(img_path, img_res)
                    else:
                        img_path = os.path.join(
                            file_dir_path, f'{i}_no_{label}.dcm')
                        img_res = np.stack(
                            [(region_mask/255)*img1, (region_mask/255)*img2], axis=0).astype('uint16')

                        img_res = sitk.GetImageFromArray(img_res)
                        img_res.SetSpacing(spacing)
                        sitk.WriteImage(img_res, img_path)

                    continue

                img1_o, img2_o, sp_img, sp_mask = ans_list
                t_num += 1
                if not USE_DCM:
                    img_path = os.path.join(
                        file_dir_path, f'{i}_yes_{label}.jpg')
                    img_res = np.concatenate(ans_list, axis=1)
                    cv2.imwrite(img_path, img_res)
                else:
                    img_path = os.path.join(
                        file_dir_path, f'{i}_yes_{label}.dcm')
                    img_res = np.stack(ans_list, axis=0).astype('uint16')
                    img_res = sitk.GetImageFromArray(img_res)
                    img_res.SetSpacing(spacing)
                    sitk.WriteImage(img_res, img_path)

            with open(os.path.join(TARGET_DIR_PATH, './logger.txt'), 'a') as f:
                f.write(
                    f'file name in this is {file_name}, t_num is {t_num}, f_num is {f_num}\n')
    print(f't_num is {t_num},f_num is {f_num}')


if __name__ == '__main__':

    deep2_filp()
    # test_prepocess_after()
    # cluster_all()
    # test_plt()
    # deal_flip()

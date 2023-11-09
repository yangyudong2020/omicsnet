# -*- coding:utf-8 _*-
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


# import
OUR_DATA_PATH = './real_my_data/'
os.makedirs(OUR_DATA_PATH, exist_ok=True)


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
all_extractor = featureextractor.RadiomicsFeatureExtractor(params)
NUM_PROCESSOR = 104


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

    return np.array(res, dtype=np.float32)


def test_prepocess_after():
    img = np.random.rand(1024, 256)
    mask = np.zeros((1024, 256))
    for i in range(300, 400, 1):
        for j in range(100, 200, 1):
            mask[i, j] = 1
    res = prepocess_after(img, mask, True)
    print(res.shape)


def work_after_cluster(div_results, TRAIN_DIR):

    print(f'now work after cluster and calculate texture')

    for ind, cls in enumerate(div_results):
        results = []
        lda_labels = []
        pool = mp.Pool(processes=NUM_PROCESSOR)
        for img, mask, label, name in tqdm(cls):
            lda_labels.append(label)
            result = pool.apply_async(
                prepocess_after, args=(img, mask))
            results.append(result)

        pool.close()
        pool.join()
        features = [r.get() for r in results]
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

        kmeans = KMeans(n_clusters=5, max_iter=100, init='k-means++')
        ans_cluster = kmeans.fit_predict(features)

        with open(os.path.join(TRAIN_DIR, f'{ind}_lda_model.pkl'), 'wb') as f:
            pickle.dump(lda, f)
        with open(os.path.join(TRAIN_DIR, f'{ind}_kmeans_model.pkl'), 'wb') as f:
            pickle.dump(kmeans, f)

        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        markers = ['o', 's', '^', 'd', 'x']
        categories = ['A', 'B', 'C', 'D', 'E']
        fig = plt.figure()
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        ans_num = np.zeros((5, 5)).astype(np.uint8)
        show_3d_labels = []
        for i in range(len(x)):
            label, name = cls[i][2], cls[i][3]
            if n_components == 3:
                ax.scatter(x[i], y[i], z[i], color=colors[label],
                           marker=markers[ans_cluster[i]], label=label)
            else:
                ax.scatter(x[i], y[i], color=colors[label],
                           marker=markers[ans_cluster[i]], label=label)

            ans_num[ans_cluster[i]][label] += 1
        # plt.legend()

        plt.savefig(os.path.join(TRAIN_DIR, f'scatter_plot_{ind}.png'))
        print(f'for old cls {ind}, the final result of new cls is \n{ans_num}')
        np.savetxt(os.path.join(
            TRAIN_DIR, f"my_array_{ind}.txt"), ans_num, fmt="%d")


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
    # print(res)
    return np.array(res, dtype=np.float32)


def prepocess_all(img, mask):
    a = prepocess(np.copy(mask))
    # b = prepocess_after(img, mask)
    # return np.concatenate((a, b))
    return a


def cluster(IS_USE, NO_USE_NAME):

    SOURCE_DIR_PATH = './data_from'
    RESULT_DIR = './show_cluster'
    TRAIN_DIR = './shape_texture_2'
    USE_DCM = True
    LABEL_TO_INDEX = {
        '_background_': 0,
        'M': 1,
        'N': 2,
        'U': 3,
        'L': 4,
        'F': 5
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    all_image = []
    left_info = []
    features = []
    lda_labels = []

    for path, dir_list, file_list in os.walk(SOURCE_DIR_PATH):
        for file_name in tqdm(file_list):
            if file_name.find('.json') == -1:
                continue
            new_name = name2number(file_name)
            with open(os.path.join(SOURCE_DIR_PATH, file_name), 'r', encoding='utf-8') as f:
                s = f.read()
                all_data = json.loads(s)
            results = []

            data1, data2 = all_data['data1'], all_data['data2']

            if not USE_DCM:
                img1 = img_b64_to_arr(data1['imageData'])[:, :, 0:4]
                img1 = cv2.resize(img1, (256, 1024))[:, :, 0]
                img2 = img_b64_to_arr(data2['imageData'])[:, :, 0:4]
                img2 = cv2.resize(img2, (256, 1024))[:, :, 0]
            else:
                dcm_data = sitk.ReadImage(os.path.join(path, new_name+'.dcm'))
                dcm_data = sitk.GetArrayFromImage(dcm_data)
                img1, img2 = dcm_data[0], dcm_data[1]

            shapes1 = data1['shapes']
            for i, shape in enumerate(shapes1):
                label = shape['label']
                label = LABEL_TO_INDEX[label]-1
                lda_labels.append(label)
                points = shape['points']
                mask = shape_to_mask(
                    img1.shape, points, shape_type=shape.get('shape_type', None))
                region = np.where(mask)
                region_img, region_mask = work_region_image(img1, region)
                all_image.append(region_img)
                left_info.append([label, file_name, region_mask])
                # features.append(prepocess(region_mask, IS_USE))
                ans = prepocess_all(region_img, region_mask)
                results.append(ans)

            shapes2 = data2['shapes']
            for i, shape in enumerate(shapes2):
                label = shape['label']
                label = LABEL_TO_INDEX[label]-1
                lda_labels.append(label)
                points = shape['points']
                mask = shape_to_mask(
                    img2.shape[:2], points, shape_type=shape.get('shape_type', None))
                region = np.where(mask)
                region_img, region_mask = work_region_image(img2, region)
                all_image.append(region_img)
                left_info.append([label, file_name, region_mask])
                # features.append(prepocess(region_mask, IS_USE))
                ans = prepocess_all(region_img, region_mask)
                results.append(ans)

            features.extend([r for r in results])

    features = np.array(features)
    np.savez(os.path.join(TRAIN_DIR, './first_features.npz'), data=features)
    print(f'finish load data, length is {features.shape}')
    lda_labels = np.array(lda_labels)
    print(lda_labels.shape)
    lda = LDA(n_components=3)
    features = lda.fit_transform(features, lda_labels)
    print(f'finish work for PCA and now start kmeans')

    x, y, z = features[:, 0].tolist(
    ), features[:, 1].tolist(), features[:, 2].tolist()
    # features = torch.tensor(features).cuda()
    # kmeans = KMeans(n_clusters=5, max_iter=100, mode='euclidean', verbose=1)
    # ans_cluster = kmeans.fit_predict(features).detach().cpu().numpy()
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++')
    ans_cluster = kmeans.fit_predict(features)

    with open(os.path.join(TRAIN_DIR, 'all_lda_model.pkl'), 'wb') as f:
        pickle.dump(lda, f)
    with open(os.path.join(TRAIN_DIR, 'all_kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)

    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    markers = ['o', 's', '^', 'd', 'x']
    categories = ['A', 'B', 'C', 'D', 'E']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ans_num = np.zeros((n_clusters, 5))
    show_3d_labels = []
    for i in range(len(x)):
        label, name = left_info[i][0], left_info[i][1]
        show_3d_labels.append(label)
        ax.scatter(x[i], y[i], z[i], color=colors[label], label=label)
        ans_num[ans_cluster[i]][label] += 1.0
    # plt.legend()

    plt.savefig(os.path.join(TRAIN_DIR, f'scatter_plot_all.png'))
    np.savetxt(os.path.join(TRAIN_DIR, f"my_array_all.txt"), ans_num, fmt="%d")
    print(ans_num.astype('int32'))
    np.savez(os.path.join(TRAIN_DIR, f'./show_3D_{NO_USE_NAME}.npz'), x=x, y=y, z=z, ans_cluster=ans_cluster,
             labels=np.array(show_3d_labels, dtype='uint8'), ans_num=ans_num)

    div_results = []
    for i in range(5):
        div_results.append([])

    for j, img in enumerate(all_image):
        label, name, region_mask, cluster = left_info[j][0], left_info[j][1], left_info[j][2], ans_cluster[j]
        div_results[cluster].append([img, region_mask, label, name])
        label_dir = os.path.join(RESULT_DIR, str(label))
        os.makedirs(label_dir, exist_ok=True)

        cluster_dir = os.path.join(label_dir, str(ans_cluster[j]))
        os.makedirs(cluster_dir, exist_ok=True)

        name_dir = os.path.join(cluster_dir, name)
        os.makedirs(name_dir, exist_ok=True)

        cv2.imwrite(os.path.join(name_dir, str(
            j)+'.jpg'), img.astype('uint8'))

    div_results = sorted(div_results, key=len)
    work_after_cluster(div_results, TRAIN_DIR)


def cluster_all():
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
    NO_USE_NAME = ''
    cluster(IS_USE, NO_USE_NAME)
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

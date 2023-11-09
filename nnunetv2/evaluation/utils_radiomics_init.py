from time import sleep
from medpy import metric
import SimpleITK as sitk
import radiomics.featureextractor as featureextractor
import radiomics
import pickle
import cv2
import gc
from tqdm import tqdm
from numba import jit
import multiprocessing as mp
import os
import sys
import numpy as np
import logging
from nnunetv2.configuration import USE_RADIOMICS

# from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

radiomics.logger.setLevel(logging.NOTSET)
TRAIN_DIR = './weights/4cls_train_all'
params = os.path.join(TRAIN_DIR, 'CT3D.yaml')
all_extractor = featureextractor.RadiomicsFeatureExtractor(params)
NUM_CLASSES = 4
NUM_SHAPE_CLUSTER = 4
NUM_CLUSTER = 4
NUMBER_USE_WEIGHT = -1
USE_POOL = True


IS_USE = {
    'x': True,  # 圆心的坐标
    'y': True,
    'w': False,
    'h': False,
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
    'e_max_axis': False,
    'e_min_axis': False,
    'e_angle': False,
    'angle': False,
    'curvature': False,
    'eccentricity': False,
    'circular_area': False,
    'circularity': False,
    'circular_perimeter': False,
    'equidiameter': False,
}
LDA_LIST, KMEANS_LIST = [], []
CLASS_ANS = np.zeros((NUM_SHAPE_CLUSTER, NUM_CLUSTER))
# tmp = ['all', '0', '1', '2', '3', '4']
tmp = ['all', '0', '1', '2', '3', '4', '5', '6', '7']
OPEN_LIST = []
for i in range(NUM_SHAPE_CLUSTER+1):
    OPEN_LIST.append(tmp[i])
for i, x in enumerate(OPEN_LIST):
    with open(os.path.join(TRAIN_DIR, f'{x}_lda_model.pkl'), 'rb') as f:
        lda = pickle.load(f)
    with open(os.path.join(TRAIN_DIR, f'{x}_kmeans_model.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    metrix = np.loadtxt(os.path.join(TRAIN_DIR, f'my_array_{x}.txt'))
    LDA_LIST.append(lda)
    if i != 0:
        CLASS_ANS[i-1] = np.argmax(metrix, axis=1)+1

    KMEANS_LIST.append(kmeans)


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
    # print(len(res))

    return np.array(res, dtype=np.float32)


def prepocess(img, contours_ans):
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
    [vx, vy, x, y] = cv2.fitLine(contours_ans, cv2.DIST_L2, 0, 0.01, 0.01)
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
            if np.isnan(eval(k)).any() or np.isinf(eval(k)).any():
                res.append(0)
            else:
                res.append(eval(k))

    # res = [x, y, w, h, area, perimeter, aspect_ratio,
    #        rect_area, extent, equi_diameter, e_center_x, e_center_y, e_max_axis, e_min_axis, e_angle, angle, curvature, eccentricity, circular_area, circularity, circular_perimeter, equidiameter]
    # print(res)
    # res += img.astype('float32').reshape(-1).tolist()
    # print(res)
    return np.array(res, dtype='float32')


def get_features(lda_list, kmeans_list, region_img, region_mask, contour):
    # cv2.imwrite('error_img.jpg', region_img)
    # cv2.imwrite('error_mask.jpg', region_mask)
    features = prepocess(np.copy(region_mask), contour).reshape(1, -1)
    features1 = lda_list[0].transform(features)
    ans1 = kmeans_list[0].predict(features1)[0]
    features = prepocess_after(region_img, region_mask).reshape(1, -1)
    features2 = lda_list[ans1+1].transform(features)
    ans2 = kmeans_list[ans1+1].predict(features2)[0]
    return ans1, ans2


def get_class(lda_list, kmeans_list, class_ans, region_img, region_mask, contour):

    try:
        ans1, ans2 = get_features(lda_list, kmeans_list,
                                  region_img, region_mask*255, contour)
    except Exception as e:
        cv2.imwrite('./error_mask.jpg',
                    (255*region_mask).astype('uint8'))
        cv2.imwrite('./error_img.jpg', region_img)
        print(f'error is {str(e)}')
        ans1, ans2 = -1, -1
    # ans1, ans2 = get_features(lda_list, kmeans_list,
    #                           region_img, region_mask*255, contour)
    if ans1 == -1:
        return 0
    return class_ans[ans1, ans2]


def calcu_seg_pred(seg_pred, img_origin, probs, show_path = None):
    probs = probs.max(axis=0)
    seg_pred, img_origin, probs = seg_pred.squeeze(
    ), img_origin.squeeze(), probs.squeeze()

    if USE_RADIOMICS:
        results, mask_list, is_left = calculate_class(((seg_pred != 0)*1).astype('uint8'), img_origin)
        mask = np.zeros_like(seg_pred)
        for j, label in enumerate(results[i]):
            ans = ((mask_list[i][j])*label).astype('int32')
            if is_left[i][j]:
                mask += np.concatenate([ans, np.zeros_like(ans)], axis=1)
            else:
                mask += np.concatenate([np.zeros_like(ans), ans], axis=1)

        mask = merge(mask, probs, seg_pred)
    else:
        mask = seg_pred
    return mask[np.newaxis, np.newaxis, ...]
    # im_show = np.concatenate([ch(mask), ch(label_alls[i])], axis=1)
    # cv2.imwrite(os.path.join(IMAGE_PATH, f'image_{i}.jpg'), im_show)


def calculate_class(lbl, images):
    results_pool_work = []
    mask_list = []
    is_left = []
    img1, img2 = images[:, 0:256], images[:, 256:512]
    lbl1, lbl2 = lbl[:, 0:256], lbl[:, 256:512]

    # ------work for lbl1--------------
    ret, thresh = cv2.threshold(
        (lbl1*255).astype('uint8'), 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) < 3:continue
        contour_mask = np.zeros_like(lbl1)        
        cv2.drawContours(contour_mask, [contour], 0, 1, -1)
        num_t = np.count_nonzero((contour_mask.astype('uint8') == 1)*1)
        if num_t <= 10:
            continue
        num_t = np.unique(contour_mask*img1)
        if len(num_t) == 1:
            continue
        results_pool_work.append(
            [contour_mask*img1, contour_mask.copy(), contour])
        mask_list.append(contour_mask)
        is_left.append(True)
        # cv2.imwrite('./test_show.jpg', (contour_mask*255).astype('uint8'))
    # ------work for lbl2--------------
    ret, thresh = cv2.threshold(
        (lbl2*255).astype('uint8'), 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) < 3:continue
        contour_mask = np.zeros_like(lbl2)
        cv2.drawContours(contour_mask, [contour], 0, 1, -1)
        num_t = np.count_nonzero((contour_mask.astype('uint8') == 1)*1)
        if num_t <= 10:
            continue
        num_t = np.unique(contour_mask*img2)
        if len(num_t) == 1:
            continue
        results_pool_work.append(
            [contour_mask*img2, contour_mask.copy(), contour])
        mask_list.append(contour_mask)
        is_left.append(False)
    results_pool = []
    global KMEANS_LIST, LDA_LIST, CLASS_ANS
    print(f'num work for this case is {len(results_pool_work)}')
    for args in results_pool_work:
        mask_img, contour_mask, contour = args
        gc.collect()
        results_pool.append(get_class(LDA_LIST, KMEANS_LIST,
                            CLASS_ANS, mask_img, contour_mask, contour))

    return results_pool, mask_list, is_left


STATIC_COLORS = np.array([
    np.array([0, 0, 0]),
    np.array([255, 0, 0]),
    np.array([0, 255, 0]),  # face
    np.array([0, 0, 255]),  # lb
    np.array([255, 255, 0]),   # rb
    np.array([255, 0, 255]),   # le
    np.array([0, 255, 255]),   # le
    np.array([255, 165, 0]),   # le
    np.array([128, 0, 128]),   # le
    np.array([255, 255, 255]),   # le
])


@jit(nopython=True)
def ch(img):
    img = img.astype('uint8')
    return np.array(STATIC_COLORS)[img].astype('uint8')


@jit(nopython=True)
def merge(mask, lbl_prob, lbl):
    # mask calculate by shape and texture
    # lbl_prob: predict by our model for label's prob
    # lbl: predict by our model for label
    ans = np.zeros((1024, 512)).astype('uint8')
    for i in range(1024):
        for j in range(512):
            if lbl[i, j] == mask[i, j]:
                ans[i, j] = mask[i, j]
            elif lbl[i, j] in [1, 2, 3]:
                ans[i, j] = lbl[i, j] if lbl_prob[i, j] >= 0.4 else mask[i, j]
            else:
                ans[i, j] = lbl[i, j] if lbl_prob[i, j] >= 0.5 else mask[i, j]
    return ans
# -*- coding:utf-8 _*-
# from now, we only deal data for nnUNet_raw, and feature worker will
# generate featuresTr dir
from multiprocessing import Pool
import gc
from typing import List
from time import sleep
import shutil
from tqdm import tqdm
import multiprocessing as mp
import cv2
import numpy as np
import os
import SimpleITK as sitk
import os
import radiomics.featureextractor as featureextractor
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy

# import
NUM_PROCESSOR = 24
NUM_CLASSES = 4
# SOURCE_DIR_PATH = 'Dataset005_OtherPet'
SOURCE_DIR_PATH = 'Dataset004_Metastasis'
TRAIN_DIR = '/root/nnUNet/weights/Metastasis_new'
LOGGER_PATH = f'{TRAIN_DIR}/train_cluster.txt'

params = '/root/nnUNet/CT3D.yaml'
all_extractor = featureextractor.RadiomicsFeatureExtractor(params)

def find_dict(cur, ans, pre_key=''):
    for k, v in cur.items():
        if isinstance(v, dict):
            find_dict(v, ans, pre_key+'_'+k)
        else:
            ans[pre_key+'_'+k] = v

def prepocess(img, mask):
    try:
        return prepocess_after(img, mask)
    except Exception as e:
        print(str(e))
        # if we meet some unknown error, we will set this item radiomics 
        # features to one ZERO tensor, which length is 2884.
        # since normal calculating results is a 2884 length tensor
        # , which pre is texture while last 28 is shape feature
        return np.zeros((2884,),dtype='float32')

def calc_entropy(y):
    """计算给定分布的熵"""
    distribution = np.bincount(y) / float(len(y))
    return entropy(distribution, base=2)

def calc_conditional_entropy(X, y):
    """计算条件熵"""
    # 求类标签的总熵
    total_entropy = calc_entropy(y)
    # 初始化条件熵
    conditional_entropies = np.zeros(X.shape[1])
    
    # 对每个特征，计算条件熵
    for i, col in enumerate(X.T):
        feature_entropy = 0
        # 对每个特征值，计算条件熵
        values, counts = np.unique(col, return_counts=True)
        for value, count in zip(values, counts):
            value_prob = count / float(len(col))
            subset_entropy = calc_entropy(y[col == value])
            feature_entropy += value_prob * subset_entropy
        conditional_entropies[i] = feature_entropy
    
    # 信息增益 = 总熵 - 条件熵
    information_gain = total_entropy - conditional_entropies
    return information_gain

def calc_intrinsic_value(X):
    """计算特征的内在价值。"""
    intrinsic_values = np.zeros(X.shape[1])
    
    for i, col in enumerate(X.T):
        if np.issubdtype(col.dtype, np.integer):
            counts = np.bincount(col)
        else:
            # 假设数据是分类的，但表示为浮点数，
            # 我们需要将其转换为整数。
            unique = np.unique(col)
            col = np.searchsorted(unique, col).astype(np.int64)
            counts = np.bincount(col)
        
        # 确保我们在计数中添加一个非常小的值来避免在熵计算中出现除以零的情况。
        counts = counts + np.finfo(np.float32).eps
        intrinsic_values[i] = entropy(counts / float(len(col)), base=2)
    
    return intrinsic_values


def calc_information_gain_ratio(X, y):
    """计算信息增益率"""
    information_gain = calc_conditional_entropy(X, y)
    intrinsic_value = calc_intrinsic_value(X)
    
    # 防止分母为0
    with np.errstate(divide='ignore', invalid='ignore'):
        information_gain_ratio = np.true_divide(information_gain, intrinsic_value)
        information_gain_ratio[~np.isfinite(information_gain_ratio)] = 0  # 无穷大或NaN设为0
    
    return information_gain_ratio

def feature_selection_by_igr(X, y, threshold=0.1):
    """根据信息增益率阈值选择特征"""
    igr = calc_information_gain_ratio(X, y)
    selected_features = igr >= threshold
    return X[:, selected_features], selected_features


def calc_information_gain(X, y):
    mutual_info = mutual_info_classif(X, y)
    all_entropy = entropy(np.bincount(y) / float(len(y)), base=2)
    information_gain = all_entropy - mutual_info
    return information_gain


def calc_information_gain_ratio(X, y):
    information_gain = calc_information_gain(X, y)
    # 计算每个特征的内在价值
    intrinsic_value = np.apply_along_axis(lambda x: entropy(x + np.finfo(np.float32).eps, base=2), 0, X)  # 防止零频率
    information_gain_ratio = information_gain / intrinsic_value
    return information_gain_ratio


# def feature_selection_by_igr(X, y, threshold=0.1):
#     igr = calc_information_gain_ratio(X, y)
#     selected_features = igr >= threshold
#     return X[:, selected_features], selected_features

    
def prepocess_after(img, mask):

    # !!!
    # normal calculating results is a 2884 length tensor
    # , which pre is texture while last 28 is shape feature
    rows, cols = np.where(mask == 1)

    # 计算坐标的均值
    x = np.mean(rows)
    y = np.mean(cols)

    data_spacing = [1, 1, 1]

    # print('img and mask', img.shape, mask.shape, np.unique(img), np.unique(mask))
    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetSpacing((float(data_spacing[0]), float(
        data_spacing[1]), float(data_spacing[2])))
    sitk_img = sitk.JoinSeries(sitk_img)
    sitk_mask = sitk.GetImageFromArray(mask)
    sitk_mask.SetSpacing((float(data_spacing[0]), float(
        data_spacing[1]), float(data_spacing[2])))
    sitk_mask = sitk.JoinSeries(sitk_mask)
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32)
    sitk_mask_array = sitk.GetArrayFromImage(sitk_mask)
    # print(f'mask unique is {np.unique(sitk_mask_array)}')
    features = all_extractor.execute(
        sitk_img, sitk_mask, label=1, voxelBased=True)

    res_features = {}
    res_shape = [y, x]
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
    print('res_value len is', len(res_value), 'res_shape len is', len(res_shape))
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
    # div_results = div_results[:20]
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
                prepocess, ((img, mask),))
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
    
    # feature number of cluser
    total_feature_count = 0

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
    print(res_value_list.shape)
    print(lda_labels.shape)
    X = res_value_list  # 假设这是你的特征矩阵
    y = lda_labels  # 假设这是你的标签
    threshold = 0.25 # 设置一个阈值
    X_selected, selected_features = feature_selection_by_igr(X, y, threshold)
    print(f'Selected {X_selected.shape[1]} features from {res_value_list.shape}')

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

    # for every label and every layer image 
    for label in range(1, NUM_CLASSES+1):
        if len(img.shape) == 3:
            for i, img_item in enumerate(img):
                lbl_b = (lbl[i]==label)*255
                num_t = np.count_nonzero(lbl_b)
                if num_t <= 10:
                    # print('skip this noise')
                    continue
                ret, thresh = cv2.threshold(lbl_b.astype(
                    'uint8'), 127, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                contours = sorted(contours, key=get_contour_position)

                for contour in contours:
                    contour_mask = np.zeros_like(lbl_b).astype('uint8')
                    cv2.drawContours(contour_mask, [contour], 0, 1, -1)
                    num_t = np.count_nonzero((contour_mask.astype('uint8') == 1)*1)
                    if num_t <= 10:
                        # print('skip this noise')
                        continue
                    # cv2.imwrite('./test_show.jpg', (contour_mask*255).astype('uint8'))
                    all_image.append(contour_mask*img_item)
                    left_info.append([label-1, file_name, contour_mask])
                    # block_array is a mask which is 0 or 1
        else:
            lbl_b = (lbl == label)*255
            ret, thresh = cv2.threshold(lbl_b.astype(
                'uint8'), 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contours = sorted(contours, key=get_contour_position)

            for contour in contours:
                contour_mask = np.zeros_like(lbl)
                cv2.drawContours(contour_mask, [contour], 0, 1, -1)
                num_t = np.count_nonzero((contour_mask.astype('uint8') == 1)*1)
                if num_t <= 10:
                    # print('skip this noise')
                    continue
                # cv2.imwrite('./test_show.jpg', (contour_mask*255).astype('uint8'))
                all_image.append(contour_mask*img)
                left_info.append([label-1, file_name, contour_mask])
                
    return all_image, left_info


def cluster():

    all_image = []
    left_info = []
    image_dir_path = os.environ.get(
        'nnUNet_raw')+f'/{SOURCE_DIR_PATH}/imagesTr'
    label_dir_path = image_dir_path.replace('images', 'labels')
    i = 0
    for path, dir_list, file_list in os.walk(image_dir_path):
        for file_name in tqdm(file_list):
            i += 1
            if i >= 2:
                break
            img = sitk.ReadImage(os.path.join(image_dir_path, file_name))
            img = sitk.GetArrayFromImage(img)
            lbl = sitk.ReadImage(os.path.join(
                label_dir_path, file_name.replace('_0000.nii.gz', '.nii.gz')))
            lbl = sitk.GetArrayFromImage(lbl)

            if len(np.unique(lbl)) == 0:
                continue
            all_image_item, left_info_item = get_item_image(
                img, lbl, file_name)
            all_image.extend(all_image_item)
            left_info.extend(left_info_item)

    div_results = []
    print(len(all_image))
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

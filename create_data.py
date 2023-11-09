# -*- coding:utf-8 _*-
from tqdm import tqdm
# from kmeans_pytorch import kmeans
# from fast_pytorch_kmeans import KMeans
import numpy as np
import os
import SimpleITK as sitk
import os
from numba import jit
from scipy.ndimage import zoom
import shutil
FULL_WIDTH = 384
FULL_HEIGHT = 768

def npz2nnunet():
    SOURCE_DIR_PATH = './real_my_data_small'
    FEATURES_PATH = './small_features'
    IMAGES_TR = './nnUNet_raw/Dataset002_Me/imagesTr'
    LABELS_TR = './nnUNet_raw/Dataset002_Me/labelsTr'
    FEATURES_TR = './nnUNet_raw/Dataset002_Me/featuresTr'
    os.makedirs(IMAGES_TR, exist_ok=True)
    os.makedirs(LABELS_TR, exist_ok=True)
    os.makedirs(FEATURES_TR, exist_ok=True)
    os.system(f'rm -rf {IMAGES_TR}/*')
    os.system(f'rm -rf {LABELS_TR}/*')
    os.system(f'rm -rf {FEATURES_TR}/*')
    num = 1
    max_len = 110
    for path, dir_list, file_list in os.walk(SOURCE_DIR_PATH):
        for file_name in tqdm(file_list):
            if file_name.find('.json') != -1:
                continue
            data = np.load(os.path.join(path, file_name))
            img, label_a = data['image'].astype(
                'int32'), data['label_a'].astype('int32')
            fea_name = file_name.split('.')[0]+'_radio.npy'
            feature = np.load(f'{FEATURES_PATH}/{fea_name}')
            max_len = max(max_len, feature.shape[0])
            feature = np.pad(feature, ((
                0, max_len - feature.shape[0]), (0, 0)), 'constant')
            # print('data is', np.unique(data), 'feature is', np.unique(feature))
            print(img.shape, label_a.shape, feature.shape)
            img = sitk.GetImageFromArray(img)
            sitk.WriteImage(img, f'{IMAGES_TR}/me_{num:04}_0000.nii.gz')
            label_a = sitk.GetImageFromArray(label_a)
            sitk.WriteImage(label_a, f'{LABELS_TR}/me_{num:04}.nii.gz')
            np.save(f'{FEATURES_TR}/me_{num:04}_0000.npy', feature)
            num += 1
    # print('max is', max_len)

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

def ch(img):
    img = img.astype('uint8')
    return np.array(STATIC_COLORS)[img].astype('uint8')

def full_features_same():
    max_len = 110
    SOURCE_DIR_PATH = './nnUNet_raw/Dataset004_Metastasis/featuresTr'
    TARGET_DIR_PATH = SOURCE_DIR_PATH.replace('featuresTr', 'featuresTr_new')
    os.makedirs(TARGET_DIR_PATH, exist_ok=True)
    for path, dir_list, file_list in os.walk(SOURCE_DIR_PATH):
        for file_name in tqdm(file_list):
            data_feature = np.load(os.path.join(path, file_name))
            # print('data shape is', data_feature.shape)
            data_feature = np.pad(data_feature, ((
                0, max_len - data_feature.shape[0]), (0, 0)), 'constant')
            # print('data shape of features', data_feature.shape)
            np.save(os.path.join(TARGET_DIR_PATH, file_name), data_feature)

    
def create_small():
    SOURCE_DIR_PATH = './real_my_data'
    SMALL_DIR_PATH = './real_my_data_small'

    os.system(f'rm -rf {SMALL_DIR_PATH}/*')
    for path, dir_list, file_list in os.walk(SOURCE_DIR_PATH):
        for file_name in tqdm(file_list):
            if file_name.find('.json') != -1:
                continue
            data = np.load(os.path.join(path, file_name))
            img, lbl = data['image'].astype(
                'int32'), data['label_a'].astype('int32')
            img, lbl = zoom(img, (FULL_HEIGHT/img.shape[0], FULL_WIDTH/img.shape[1]), order=3).astype(
                'int32'), zoom(lbl, (FULL_HEIGHT/lbl.shape[0], FULL_WIDTH/lbl.shape[1]), order=0).astype('uint8')
            # cv2.imwrite('results.jpg', ch(lbl))
            # print('pressed any key continue')
            # a = input()
            np.savez(f'{SMALL_DIR_PATH}/{file_name}', image = img, label_a = lbl)


if __name__ == '__main__':
    # npz2nnunet()
    # create_small()
    full_features_same()
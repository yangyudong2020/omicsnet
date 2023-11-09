import os

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

# default_num_processes = 8 if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])

ANISO_THRESHOLD = 3  # determines when a sample is considered anisotropic (3 means that the spacing in the low
# resolution axis must be 3x as large as the next largest spacing)

default_n_proc_DA = get_allowed_n_proc_DA()

default_num_processes = 32 # 使用32个进程进行nnUNet的预处理数据、使用32个进程加载要推理的数据 
default_n_proc_DA = 32 # 应该是没用的 
trainloader_num_processes = 24 # 使用32个进程进行加载训练数据 
num_iterations_per_epoch = 20 # 每一轮迭代20次，意味着如果batchsize等于10，则每一个epoch会训练200个样本。下一epoch读取样本会在上个epoch的基础上继续 
num_epochs = 1000 
num_valid = 25 # 多少epoch间隔，进行一次test 
start_num_valid = 200 # 多少轮以后开始测试 
initial_learning_rate = 2e-4

USE_RADIOMICS = False # 是否使用组学计算 
NUM_PROCESSOR_RADIO = 72 # 使用多少个进程进行组学计算 only work for USE_RADIOMICS = False

FULL_HEIGHT, FULL_WIDTH = 768, 384 # 图片的高和宽
# FULL_HEIGHT, FULL_WIDTH = 1024, 512

USING_UNETRPP = True 
USING_3D_FOR_UNETRPP = False # only work for USING_UNETRPP = True 
USING_CLIP = True # only work for USING_UNETRPP = True 
USING_ADJUST_LOSS = False
# dataset_num = '005_OtherPet' # 数据集名称，一定要写对, only need to contain number and name
# dataset_num = '006_Kidney'
dataset_num = '004_Metastasis'

clip_content=(2884, 110)
IS_COMPARING = False # this is the first chose since if it is chosen we will compare model first
COMPARING_MODEL_TYPE = 'unet' # only work for IS_COMPARING = True
# we can choose 'unetr missformer swinunet transunet unet' for 2D metastasis compare experiment
# for unet, we need to choose 640 384 as input size, while other must choose 224 224!
# Attention:
# we only use evaluate_predictions.py and utils_radiomics_small.py both of them work for 2D
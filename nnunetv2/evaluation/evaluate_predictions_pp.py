from medpy import metric
from nnunetv2.evaluation.utils_radiomics_pp import *
import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        splitted = key.split(',')
        return tuple([int(i) for i in splitted])


def save_summary_json(results: dict, output_file: str):
    """
    stupid json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(
        k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    results['mean'] = {key_to_label_or_region(
        k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results["metric_per_case"])):
        results["metric_per_case"][i]['metrics'] = \
            {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def calculate_all(prediction_probs, predictions, images, label_alls, reference_files, prediction_files, labels_or_regions):
    if USE_RADIOMICS:
        results_list, mask_list, is_left = calculate_class(
            ((predictions != 0)*1).astype('uint8'), images)
        print(f'prediction_probs is {len(prediction_probs)}')
    results = []
    for i, lbl_probs in enumerate(prediction_probs):
        results.append({})
        if USE_RADIOMICS:
            mask = np.zeros_like(lbl_probs)
            for j, label in enumerate(results_list[i]):
                ans = ((mask_list[i][j])*label).astype('int32')
                if is_left[i][j]:
                    mask += np.concatenate([ans, np.zeros_like(ans)], axis=1)
                else:
                    mask += np.concatenate([np.zeros_like(ans), ans], axis=1)

            mask = merge(mask, lbl_probs, predictions[i])
        else:
            mask = predictions[i]
        
        im_show = np.concatenate([ch(mask), ch(label_alls[i])], axis=1)

        results[i]['reference_file'] = reference_files[i]
        results[i]['prediction_file'] = prediction_files[i]
        results[i]['metrics'] = {}
        for r in labels_or_regions:
                results[i]['metrics'][r] = {}
                seg_ref = label_alls[i][np.newaxis,np.newaxis,...]
                seg_pred = mask[np.newaxis,np.newaxis,...]
                mask_ref = region_or_label_to_mask(seg_ref, r)
                mask_pred = region_or_label_to_mask(seg_pred, r)
                tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, None)
                if tp + fp + fn == 0:
                    results[i]['metrics'][r]['Dice'] = np.nan
                    results[i]['metrics'][r]['IoU'] = np.nan
                else:
                    results[i]['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
                    results[i]['metrics'][r]['IoU'] = tp / (tp + fp + fn)
                results[i]['metrics'][r]['FP'] = fp
                results[i]['metrics'][r]['TP'] = tp
                results[i]['metrics'][r]['FN'] = fn
                results[i]['metrics'][r]['TN'] = tn
                results[i]['metrics'][r]['n_pred'] = fp + tp
                results[i]['metrics'][r]['n_ref'] = fn + tp
                results[i]['metrics'][r]['precision'] = tp / \
                    (tp + fp) if tp + fp > 0 else np.nan
                results[i]['metrics'][r]['recall'] = tp / \
                    (tp + fn) if tp + fn > 0 else np.nan
                results[i]['metrics'][r]['hd95'] = metric.binary.hd95(
                    mask_pred, mask_ref) if mask_pred.sum() > 0 and mask_ref.sum() > 0 else np.nan
        # cv2.imwrite(os.path.join(IMAGE_PATH, f'image_{i}.jpg'), im_show)
        # for j in range(1, NUM_CLASSES+1):
        #     metric_list_all[j-1].append(calculate_metric_percase(
        #         mask == j, label_alls[i] == j))

    # metric_list_all = np.array(metric_list_all)
    return results


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith(
            '.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_pred exist in folder_ref"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    num_processes = min(num_processes, len(files_pred))

    predictions = []
    prediction_probs = []
    images = []
    label_alls = []
    for i, (reference_file, prediction_file) in enumerate(zip(files_ref, files_pred)):
        seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
        seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
        img_origin_file = os.path.join('./nnUNet_raw/Dataset001_Me/imagesTr',
                                    prediction_file.split('/')[-1].replace('.nii.gz', '_0000.nii.gz'))
        img_origin, img_origin_dict = image_reader_writer.read_seg(img_origin_file)
        # spacing = seg_ref_dict['spacing']
        img_origin = img_origin.astype('int32')
        probs_file = prediction_file.replace('.nii.gz', '.npy')
        probs = np.load(probs_file).max(axis=0)

        predictions.append(seg_pred.squeeze(axis=0))
        prediction_probs.append(probs)
        images.append(img_origin.squeeze(axis=0))
        label_alls.append(seg_ref.squeeze(axis=0))

    predictions = np.concatenate(predictions, axis=0)
    prediction_probs = np.concatenate(prediction_probs, axis=0)
    images = np.concatenate(images, axis=0)
    label_alls = np.concatenate(label_alls, axis=0)
    print('shape is predictions, prediction_probs, images, label_alls', predictions.shape, prediction_probs.shape, images.shape, label_alls.shape)


    results = calculate_all(prediction_probs, predictions, images, label_alls, files_ref, files_pred, regions_or_labels)

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means,
              'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result
    # print('DONE')


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(
        dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill)


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None,
                                     num_processes: int = default_num_processes,
                                     ignore_label: int = None,
                                     chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
                                                  verbose=False)()
    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str,
                        help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str,
                        help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true',
                        help='dont crash if folder_pred doesnt have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder,
                               args.djfile, args.pfile, args.o, args.np, chill=args.chill)


def evaluate_simple_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str,
                        help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str,
                        help='folder with predicted segmentations')
    parser.add_argument('-l', type=int, nargs='+', required=True,
                        help='list of labels')
    parser.add_argument('-il', type=int, required=False, default=None,
                        help='ignore label')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true',
                        help='dont crash if folder_pred doesnt have all files that are present in folder_gt')

    args = parser.parse_args()
    compute_metrics_on_folder_simple(
        args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il, chill=args.chill)


if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions, ignore_label,
                              num_processes)

import os
import json
import numpy as np
from .sitk_stuff import read_nifti
from .writer import write_mha
from .json_pickle_stuff import read_pickle
from .paths_dirs_stuff import path_contents_pattern, create_path


def run_fullres(out_path, path_prepared_data, nnunet_path_out):
    save_mha_path = os.path.join(out_path, "images/automated-petct-lesion-segmentation")
    create_path(save_mha_path)
    crop_path_log = os.path.join(path_prepared_data, 'crop_log')
    cropped_pred_files = path_contents_pattern(nnunet_path_out, '.nii.gz')
    cropped_log_files = path_contents_pattern(crop_path_log, '.pkl')
    n_subject = len(cropped_pred_files)

    for ix, file in enumerate(cropped_pred_files):
        print('\t'*2, 'projecting full-res mask {} out of {} ...'.format(ix + 1, n_subject))
        subject_name = file.split('.nii.gz')[0]
        crop_seg_path = os.path.join(nnunet_path_out, file)
        seg_array, seg_itk, _, seg_spacing, seg_origin, seg_direction = read_nifti(crop_seg_path)

        log_name = [x for x in cropped_log_files if subject_name in x][0]
        orig_file_name = log_name.split('.pkl')[0]
        log_path = os.path.join(crop_path_log, log_name)
        log_dict = read_pickle(log_path)

        orig_img_size = log_dict['orig_array_size']
        seg_mask_fullsize = np.zeros(orig_img_size, dtype='uint8')
        z_start = log_dict['z_start']
        z_end = log_dict['z_end']
        y_start = log_dict['y_start']
        y_end = log_dict['y_end']
        x_start = log_dict['x_start']
        x_end = log_dict['x_end']

        seg_mask_fullsize[z_start:z_end, y_start:y_end, x_start:x_end] = seg_array
        orig_mha_filename = orig_file_name+'.mha'
        write_name = os.path.join(save_mha_path, orig_mha_filename)
        write_mha(seg_mask_fullsize, seg_spacing, seg_origin, seg_direction, write_name)

    return None

def save_datacentric(out_path, value: bool):
    dc_json_path = os.path.join(out_path, "data-centric-model.json")
    with open(dc_json_path, "w") as json_file:
        json.dump(value, json_file, indent=4)
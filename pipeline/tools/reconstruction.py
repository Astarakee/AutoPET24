import os
import numpy as np
from tools.sitk_stuff import read_nifti
from tools.writer import write_nifti_from_vol
from tools.json_pickle_stuff import read_pickle
from tools.paths_dirs_stuff import path_contents_pattern, create_path


def run_fullres(out_path):
    pred_crop_out = os.path.join(out_path, 'pred_crops')
    cop_path_log = os.path.join(out_path, 'crop_log')
    pref_fullres_out = os.path.join(out_path, 'predictions')
    create_path(pref_fullres_out)
    #
    cropped_pred_files = path_contents_pattern(pred_crop_out, '.nii.gz')
    cropped_log_files = path_contents_pattern(cop_path_log, '.pkl')
    n_subject = len(cropped_pred_files)

    for ix, file in enumerate(cropped_pred_files):
        print('\t'*2, 'projecting full-res mask {} out of {} ...'.format(ix + 1, n_subject))
        subject_name = file.split('.nii.gz')[0]
        crop_seg_path = os.path.join(pred_crop_out, file)
        seg_array, seg_itk, _, seg_spacing, seg_origin, seg_direction = read_nifti(crop_seg_path)

        log_name = [x for x in cropped_log_files if subject_name in x.replace(' ', '')][0]
        orig_file_name = log_name.split('.pkl')[0]
        log_path = os.path.join(cop_path_log, log_name)
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
        orig_nifti_name = orig_file_name+'.nii.gz'
        write_name = os.path.join(pref_fullres_out, orig_nifti_name)
        write_nifti_from_vol(seg_mask_fullsize, seg_origin, seg_spacing, seg_direction, write_name)

    return None

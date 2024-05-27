import os
import numpy as np
import SimpleITK as itk
from tools.sitk_stuff import read_nifti
from tools.writer import write_nifti_from_vol
from tools.json_pickle_stuff import write_pickle, read_pickle
from tools.paths_dirs_stuff import path_contents_pattern, path_contents, create_path

crop_pred_path = '/home/mehdi/Data/Dataset705_AutoPET24Crop/labelsTr'
crop_log_path = '/home/mehdi/Data/Dataset705_AutoPET24Crop/crop_log'
fullsize_seg_path = '/home/mehdi/Data/Dataset705_AutoPET24Crop/full_size_mask_tst'
create_path(fullsize_seg_path)
#
cropped_pred = path_contents_pattern(crop_pred_path, '.nii.gz')
cropped_logs = path_contents_pattern(crop_log_path, '.pkl')
n_subject = len(cropped_pred)

for ix, file in enumerate(cropped_pred):
    print('Projecting cropped mask into full res mask: subject {} out of {} in progress ...'.format(ix + 1, n_subject))

    subject_name = file.split('.nii.gz')[0]
    crop_seg_path = os.path.join(crop_pred_path, file)

    seg_array, seg_itk, _, seg_spacing, seg_origin, seg_direction = read_nifti(crop_seg_path)

    log_name = [x for x in cropped_logs if subject_name in x][0]
    log_path = os.path.join(crop_log_path, log_name)
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

    write_name = os.path.join(fullsize_seg_path, file)
    write_nifti_from_vol(seg_mask_fullsize, seg_origin, seg_spacing, seg_direction, write_name)
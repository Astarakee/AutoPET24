import os
import numpy as np
import SimpleITK as itk
from copy import deepcopy
from tools.sitk_stuff import read_nifti
from tools.writer import write_nifti_from_vol
from tools.json_pickle_stuff import write_pickle
from tools.croping_stuff import bbox_coordinate, creat_bbox
from tools.paths_dirs_stuff import path_contents_pattern, create_path



def windowing_intensity(img_array, min_bound, max_bound):
    
    img_array[img_array>max_bound] = max_bound
    img_array[img_array<min_bound] = min_bound
    return img_array


def run_prepare(in_path, out_path):
    nnunet_in_path = os.path.join(out_path, 'imagesTr')
    cop_path_log = os.path.join(out_path, 'crop_log')

    create_path(nnunet_in_path)
    create_path(cop_path_log)
    ct_files = path_contents_pattern(in_path, '_0000.nii.gz')
    pt_files = path_contents_pattern(in_path, '_0001.nii.gz')

    n_subject = len(ct_files)
    xy_margin = 3
    min_ct_int = -800
    max_ct_int = 800
    for ix, _ in enumerate(ct_files):
        print('\t'*2, 'preparing case {} out of {}'.format(ix + 1, n_subject))
        case_ct = ct_files[ix]
        case_pt = pt_files[ix]
        subject_name = case_ct.split('_0000.nii.gz')[0]
        case_path_ct = os.path.join(in_path, case_ct)
        case_path_pt = os.path.join(in_path, case_pt)

        ct_array, ct_itk, ct_size, ct_spacing, ct_origin, ct_direction = read_nifti(case_path_ct)
        pt_array, _, _, _, _, _ = read_nifti(case_path_pt)
        array_size = ct_array.shape

        # img_binary = windowing_intensity(ct_array, min_ct_int, max_ct_int)
        img_binary = deepcopy(ct_array)
        img_binary[img_binary>min_ct_int] = 1
        img_binary[img_binary<=min_ct_int] = 0
        img_binary = img_binary.astype(np.uint8)

        binary_itk = itk.GetImageFromArray(img_binary)
        binary_itk.SetSpacing(ct_spacing)
        binary_itk.SetOrigin(ct_origin)
        binary_itk.SetDirection(ct_direction)

        itk_array, x_start, x_end, y_start, y_end, z_start, z_end = bbox_coordinate(binary_itk, xy_margin)
        temp_new_array, x_size, y_size, z_size, x_end, y_end, z_end = creat_bbox(itk_array, x_start, x_end, y_start, y_end, z_start, z_end, ct_spacing, ct_origin, ct_direction)

        ct_array = windowing_intensity(ct_array, min_ct_int, max_ct_int)
        masked_ct = temp_new_array*ct_array
        masked_pt = temp_new_array*pt_array
        cropped_ct = np.zeros((z_size, y_size, x_size))
        cropped_pt = np.zeros((z_size, y_size, x_size))
        cropped_sg = np.zeros((z_size, y_size, x_size))
        cropped_ct = masked_ct[z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_pt = masked_pt[z_start:z_end, y_start:y_end, x_start:x_end]

        case_ct = case_ct.replace(' ', '') # white space removal
        case_pt = case_pt.replace(' ', '')
        cropped_img_path_ct = os.path.join(nnunet_in_path, case_ct)
        cropped_img_path_pt = os.path.join(nnunet_in_path, case_pt)
        crop_log_path = os.path.join(cop_path_log, subject_name+'.pkl')

        logs = {}
        logs['orig_array_size'] = array_size
        logs['z_start'] = z_start
        logs['z_end'] = z_end
        logs['y_start'] = y_start
        logs['y_end'] = y_end
        logs['x_start'] = x_start
        logs['x_end'] = x_end
        logs['orders'] = 'array[z_start:z_end, y_start:y_end, x_start:x_end]'
        write_pickle(crop_log_path, logs)
        write_nifti_from_vol(cropped_ct, ct_origin, ct_spacing, ct_direction, cropped_img_path_ct)
        write_nifti_from_vol(cropped_pt, ct_origin, ct_spacing, ct_direction, cropped_img_path_pt)

    return None
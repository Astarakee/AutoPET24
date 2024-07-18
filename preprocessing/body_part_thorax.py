import os
import numpy as np
from tools.sitk_stuff import read_nifti
from tools.writer import write_nifti_from_vol
from tools.json_pickle_stuff import write_pickle
from tools.paths_dirs_stuff import path_contents_pattern, create_path


data_path = '/home/mehdi/Data/AutoPET24/Dataset704_AutoPET24Crop'
totalseg_path = '/home/mehdi/Data/AutoPET24/Dataset704_AutoPET24Crop/total_seg'
write_path = '/home/mehdi/Data/AutoPET24/Dataset705_AutoPET24Crop_Thorax'
images_path = os.path.join(data_path, 'imagesTr')
labels_path = os.path.join(data_path, 'labelsTr')
log_path = os.path.join(write_path, 'body_logs')
write_path_images = os.path.join(write_path, 'imagesTr')
write_path_labels = os.path.join(write_path, 'labelsTr')
create_path(write_path_images)
create_path(write_path_labels)
create_path(log_path)

image_files = path_contents_pattern(images_path, '.nii.gz')
label_files = path_contents_pattern(labels_path, '.nii.gz')
n_subject = len(label_files)
axial_margin = 5

for ix, case in enumerate(label_files):
    print('-'*6)
    print('working on subject {} out of {}'.format(ix+1, n_subject))
    temp = {}
    case_name = case.split('.nii')[0]
    case_name_ct = case_name+'_0000.nii.gz'
    case_name_pt = case_name+'_0001.nii.gz'
    case_path_ct = os.path.join(images_path, case_name_ct)
    case_path_pt = os.path.join(images_path, case_name_pt)
    case_path_sg = os.path.join(labels_path, case_name)
    case_path_totalseg = os.path.join(totalseg_path, case_name)
    l_clavicle_path = os.path.join(case_path_totalseg, 'clavicula_left.nii.gz')
    r_clavicle_path = os.path.join(case_path_totalseg, 'clavicula_right.nii.gz')
    l_rib_path = os.path.join(case_path_totalseg, 'rib_left_11.nii.gz')
    r_rib_path = os.path.join(case_path_totalseg, 'rib_right_11.nii.gz')

    ct_array, ct_itk, _, ct_spacing, ct_origin, ct_direction = read_nifti(case_path_ct)
    pt_array, _, _, _, _, _ = read_nifti(case_path_pt)
    sg_array, _, _, _, _, _ = read_nifti(case_path_sg)
    l_cl_array, _, _, _, _, _ = read_nifti(l_clavicle_path)
    r_cl_array, _, _, _, _, _ = read_nifti(r_clavicle_path)
    l_rib_array, _, _, _, _, _ = read_nifti(l_rib_path)
    r_rib_array, _, _, _, _, _ = read_nifti(r_rib_path)
    array_shape = ct_array.shape
    clavicle_array = l_cl_array+r_cl_array
    rib11_array = l_rib_array+r_rib_array
    clavicle_array[clavicle_array!=0] = 1
    rib11_array[rib11_array!=0] = 1
    ax_ind_upper, _, _ = np.where(clavicle_array==1)
    ax_ind_lower, _, _ = np.where(rib11_array==1)
    min_ax_slice_lower = np.min(ax_ind_lower)
    min_ax_slice_upper = np.min(ax_ind_upper)
    min_ax_marginal_lower = min_ax_slice_lower-axial_margin
    min_ax_marginal_upper = min_ax_slice_upper+axial_margin

    hn_ct_arr = ct_array[min_ax_marginal_lower:min_ax_marginal_upper,:,:]
    hn_pt_arr = pt_array[min_ax_marginal_lower:min_ax_marginal_upper,:,:]
    hn_sg_arr = sg_array[min_ax_marginal_lower:min_ax_marginal_upper,:,:]

    ct_save_abs_path = os.path.join(write_path_images, case_name_ct)
    pt_save_abs_path = os.path.join(write_path_images, case_name_pt)
    sg_save_abs_path = os.path.join(write_path_labels, case_name)
    log_save_abs_path = os.path.join(log_path, case_name+'.pkl')
    temp['orig_data_path'] = data_path
    temp['write_path'] = write_path
    temp['subject_name'] = case_name
    temp['array_shape'] = array_shape
    temp['axis_marginal'] = axial_margin
    temp['start_slice_crop'] = min_ax_marginal_upper
    temp['end_slice_crop'] = min_ax_marginal_lower
    write_nifti_from_vol(hn_ct_arr, ct_origin, ct_spacing, ct_direction, ct_save_abs_path)
    write_nifti_from_vol(hn_pt_arr, ct_origin, ct_spacing, ct_direction, pt_save_abs_path)
    write_nifti_from_vol(hn_sg_arr, ct_origin, ct_spacing, ct_direction, sg_save_abs_path)
    write_pickle(log_save_abs_path, temp)
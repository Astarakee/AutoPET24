import os
import numpy as np
import SimpleITK as itk
from copy import deepcopy
from tools.sitk_stuff import read_nifti
from tools.writer import write_nifti_from_vol
from tools.json_pickle_stuff import write_pickle
from tools.preprocess import windowing_intensity
from tools.paths_dirs_stuff import path_contents_pattern, create_path


in_path = '/home/mehdi/Data/AutoPET'
out_path = '/home/mehdi/Data/Dataset705_AutoPET24Crop'
in_path_img = os.path.join(in_path, 'imagesTr')
in_path_mask = os.path.join(in_path, 'labelsTr')
out_path_img = os.path.join(out_path, 'imagesTr')
out_path_mask = os.path.join(out_path, 'labelsTr')
out_path_log = os.path.join(out_path, 'crop_log')
create_path(out_path_img)
create_path(out_path_mask)
create_path(out_path_log)

ct_files = path_contents_pattern(in_path_img, '_0000.nii.gz')
pt_files = path_contents_pattern(in_path_img, '_0001.nii.gz')
sg_files = path_contents_pattern(in_path_mask, '.nii.gz')

n_subject = len(sg_files)
xy_margin = 3
min_ct_int = -800
max_ct_int = 800

for ix in range(n_subject):
    print('working on case {} out of {}'.format(ix + 1, n_subject))
    case_ct = ct_files[ix]
    case_pt = pt_files[ix]
    case_sg = sg_files[ix]
    subject_name = case_sg.split('.nii.gz')[0]
    case_path_ct = os.path.join(in_path_img, case_ct)
    case_path_pt = os.path.join(in_path_img, case_pt)
    case_path_sg = os.path.join(in_path_mask, case_sg)
    ct_array, ct_itk, ct_size, ct_spacing, ct_origin, ct_direction = read_nifti(case_path_ct)
    pt_array, _, _, _, _, _ = read_nifti(case_path_pt)
    sg_array, _, _, _, _, _ = read_nifti(case_path_sg)
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

    # largets connect component contains only body
    component_image = itk.ConnectedComponent(binary_itk)
    sorted_component_image = itk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    itk_array = itk.GetArrayFromImage(largest_component_binary_image)

    # bounding box coordinates
    lsif = itk.LabelShapeStatisticsImageFilter()
    lsif.Execute(largest_component_binary_image)
    boundingBox = np.array(lsif.GetBoundingBox(1))
    x_start, y_start, z_start, x_size, y_size, z_size = boundingBox

    z_dim = itk_array.shape[0]
    x_end = x_start + x_size
    y_end = y_start + y_size
    z_end = z_dim  # get all axial slices
    z_start = 0
    # add margins for sanity
    x_start -= xy_margin
    x_end += xy_margin
    y_start -= xy_margin
    y_end += xy_margin

    # creating masks for orthogonal views
    new_array1 = np.zeros_like(itk_array)
    new_array2 = np.zeros_like(itk_array)
    new_array3 = np.zeros_like(itk_array)
    new_array1[z_start:z_end, :, :] = 10
    new_array2[:, :, x_start:x_end] = 10
    new_array3[:, y_start:y_end, :] = 10
    temp_new_array = new_array1 + new_array2 + new_array3
    # binarizing the mask
    temp_new_array[temp_new_array == 30] = 100
    temp_new_array[temp_new_array != 100] = 0
    temp_new_array[temp_new_array == 100] = 1
    temp_new_array = temp_new_array.astype('uint8')

    new_itk = itk.GetImageFromArray(temp_new_array)
    new_itk.SetSpacing(ct_spacing)
    new_itk.SetOrigin(ct_origin)
    new_itk.SetDirection(ct_direction)

    # bounding box coordinates
    lsif = itk.LabelShapeStatisticsImageFilter()
    lsif.Execute(new_itk)
    boundingBox = np.array(lsif.GetBoundingBox(1))
    x_start, y_start, z_start, x_size, y_size, z_size = boundingBox
    x_end = x_start+x_size
    y_end = y_start+y_size
    z_end = z_start+z_size

    ct_array = windowing_intensity(ct_array, min_ct_int, max_ct_int)
    masked_ct = temp_new_array*ct_array
    masked_pt = temp_new_array*pt_array
    masked_sg = temp_new_array*sg_array
    cropped_ct = np.zeros((z_size, y_size, x_size))
    cropped_pt = np.zeros((z_size, y_size, x_size))
    cropped_sg = np.zeros((z_size, y_size, x_size))
    cropped_ct = masked_ct[z_start:z_end, y_start:y_end, x_start:x_end]
    cropped_pt = masked_pt[z_start:z_end, y_start:y_end, x_start:x_end]
    cropped_sg = masked_sg[z_start:z_end, y_start:y_end, x_start:x_end]

    cropped_img_path_ct = os.path.join(out_path_img, case_ct)
    cropped_img_path_pt = os.path.join(out_path_img, case_pt)
    cropped_img_path_sg = os.path.join(out_path_mask, case_sg)
    crop_log_path = os.path.join(out_path_log, subject_name+'.pkl')

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
    write_nifti_from_vol(cropped_sg, ct_origin, ct_spacing, ct_direction, cropped_img_path_sg)
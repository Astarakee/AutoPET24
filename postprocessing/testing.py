import os
import numpy as np
import SimpleITK as itk
from tools.sitk_stuff import read_nifti
from tools.paths_dirs_stuff import path_contents_pattern


data_path = "/home/mehdi/Data/AutoPET24/Dataset704_AutoPET24Crop/"
img_path = os.path.join(data_path, "imagesTr")
label_path = os.path.join(data_path, "labelsTr")
img_files = path_contents_pattern(img_path, ".nii.gz")
label_files = path_contents_pattern(label_path, ".nii.gz")
n_subject = len(label_files)
dataset_stats = {}
label_files = label_files[:10]
for ix, case in enumerate(label_files):
    print("working on subject {} out of {}".format(ix+1, n_subject))
    tmp_stats = {}
    case_name = case.split('.nii.gz')[0]
    case_image_names = [x for x in img_files if case_name in x]
    case_ct = [x for x in case_image_names if "_0000" in x][0]
    case_pt = [x for x in case_image_names if "_0001" in x][0]

    case_sg_path = os.path.join(label_path, case)
    case_ct_path = os.path.join(img_path, case_ct)
    case_pt_path = os.path.join(img_path, case_pt)

    sg_array, sg_itk, sg_size, sg_spacing, sg_origin, sg_direction = read_nifti(case_sg_path)
    ct_array, ct_itk, _, _, _, _ = read_nifti(case_ct_path)
    pt_array, pt_itk, _, _, _, _ = read_nifti(case_pt_path)

    stats = itk.LabelShapeStatisticsImageFilter()
    component_image = itk.ConnectedComponent(sg_itk)
    stats.Execute(component_image)
    component_labels = [x for x in stats.GetLabels()]
    n_components = len(component_labels)
    tmp_stats['n_components'] = n_components
    min_ct_vals = []
    max_ct_vals = []
    min_pt_vals = []
    max_pt_vals = []
    med_ct_vals = []
    med_pt_vals = []
    region_size = []
    for label in stats.GetLabels():
        çomponent_name = 'label'+str(label)
        binary_itk = component_image == label
        binary_comp_array = itk.GetArrayFromImage(binary_itk)
        label_sizes = stats.GetNumberOfPixels(label)
        ct_target_vol = binary_comp_array * ct_array
        pt_target_vol = binary_comp_array * pt_array
        ct_non_zeros = ct_target_vol[np.where(ct_target_vol != 0)]
        pt_non_zeros = pt_target_vol[np.where(pt_target_vol != 0)]
        max_int_ct = np.max(ct_non_zeros)
        min_int_ct = np.min(ct_non_zeros)
        med_int_ct = np.median(ct_non_zeros)
        max_int_pt = np.max(pt_non_zeros)
        min_int_pt = np.min(pt_non_zeros)
        med_int_pt = np.median(pt_non_zeros)
        min_ct_vals.append(min_int_ct)
        max_ct_vals.append(max_int_ct)
        med_ct_vals.append(med_int_ct)
        min_pt_vals.append(min_int_pt)
        max_pt_vals.append(max_int_pt)
        med_pt_vals.append(med_int_pt)
        region_size.append(label_sizes)

    tmp_stats['region_size'] = region_size
    tmp_stats['min_ct_vals'] = min_ct_vals
    tmp_stats['max_ct_vals'] = max_ct_vals
    tmp_stats['med_ct_vals'] = med_ct_vals
    tmp_stats['min_pt_vals'] = min_pt_vals
    tmp_stats['max_pt_vals'] = max_pt_vals
    tmp_stats['med_pt_vals'] = med_pt_vals
    dataset_stats[case_name] = tmp_stats




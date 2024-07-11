import os
import numpy as np
import SimpleITK as itk
from tools.sitk_stuff import read_nifti
from tools.json_pickle_stuff import write_pickle
from tools.paths_dirs_stuff import path_contents_pattern


img_path = "/home/mehdi/Data/AutoPET24/Dataset704_AutoPET24Crop/imagesTr"
pred_path = "/mnt/workspace/data/nnUnet/nnUNet_results/AutoPET24/Dataset704_AutoPET24Crop/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_2/validation"
save_path = "/home/mehdi/Data/AutoPET24/postprocessing/PETRegion10MeanThr-2/fold_2"

pred_files = os.listdir(pred_path)
img_files = os.listdir(img_path)
pred_files = [x for x in pred_files if '.nii.gz' in x]
pt_files = [x for x in img_files if '_0001.nii.gz' in x]
n_subject = len(pred_files)

pt_mean_intensity_per1 = 1.4
pt_mean_intensity_mean = 5.3970
pt_mean_thr = pt_mean_intensity_mean-2

for ix, case in enumerate(pred_files):
    print("working on subject {} out of {}".format(ix+1, n_subject))
    case_name = case.split('.nii.gz')[0]
    pred_abs_path = os.path.join(pred_path, case)
    pt_img_file = [x for x in pt_files if case_name in x][0]
    pt_abs_path = os.path.join(img_path, pt_img_file)

    sg_array, sg_itk, sg_size, sg_spacing, sg_origin, sg_direction = read_nifti(pred_abs_path)
    pt_array, pt_itk, _, _, _, _ = read_nifti(pt_abs_path)

    stats = itk.LabelShapeStatisticsImageFilter()
    component_image = itk.ConnectedComponent(sg_itk)
    stats.Execute(component_image)
    component_labels = [x for x in stats.GetLabels()]
    n_components = len(component_labels)
    component_array = itk.GetArrayFromImage(component_image)
    if n_components>0:
        for label in stats.GetLabels():
            binary_itk = component_image == label
            binary_comp_array = itk.GetArrayFromImage(binary_itk)
            binary_comp_array[binary_comp_array!=0] = 1
            label_sizes = stats.GetNumberOfPixels(label)

            if label_sizes<10:
                pt_seg_img = pt_array*binary_comp_array
                pt_non_zeros = pt_seg_img[pt_seg_img!=0]
                pt_mean_intensity = np.mean(pt_non_zeros)
                pt_min_intensity = np.min(pt_non_zeros)
                pt_percentile1_intensity = np.percentile(pt_non_zeros, 1)
                # condition, remove spots with thresholding criteria:
                # if pt_mean_intensity<pt_mean_intensity_per1:
                if pt_mean_intensity < pt_mean_thr:
                    component_array[component_array==label]=0
                    print('\t'*2, 'condition apppiled!')
                else:
                    pass
            else:
                pass

        component_array[component_array!=0] = 1
        new_component_itk = itk.GetImageFromArray(component_array)
        new_component_itk.SetSpacing(sg_spacing)
        new_component_itk.SetOrigin(sg_origin)
        new_component_itk.SetDirection(sg_direction)

    else:
        new_component_itk = sg_itk

    write_path = os.path.join(save_path, case)
    itk.WriteImage(new_component_itk, write_path)
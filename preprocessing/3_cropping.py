import os
import numpy as np
import SimpleITK as itk
from tools.sitk_stuff import read_nifti
from tools.writer import write_nifti_from_vol
from tools.json_pickle_stuff import write_pickle
from tools.preprocess import windowing_intensity
from tools.paths_dirs_stuff import path_contents_pattern, path_contents, create_path

img_path = '/home/mehdi/Data/AutoPET/imagesTr'
ground_path = '/home/mehdi/Data/AutoPET/labelsTr'
mask_path = '/home/mehdi/Data/Crop_AutoPET/CropMask'
cropped_path_img = '/home/mehdi/Data/Dataset704_AutoPET24Crop/imagesTr'
cropped_path_lablel = '/home/mehdi/Data/Dataset704_AutoPET24Crop/labelsTr'
log_path = '/home/mehdi/Data/Dataset704_AutoPET24Crop/cropped_img_logs'

min_ct_int = -800
max_ct_int = 800
min_pt_int = 0
max_pt_int = 15


create_path(cropped_path_img)
create_path(cropped_path_lablel)
create_path(log_path)
img_files_ch0 = path_contents_pattern(img_path, '0000.nii.gz')
img_files_ch1 = path_contents_pattern(img_path, '0001.nii.gz')
n_img = len(img_files_ch0)

mask_subjects = path_contents(mask_path)


for ixx in range(n_img):
    print('Cropping subject {} out of {}'.format(ixx+1, n_img))
    
    img0_name = img_files_ch0[ixx]
    img1_name = img_files_ch1[ixx]
    subject_name = img0_name.split('_0000.nii.gz')[0]
    
    img0_path = os.path.join(img_path, img0_name)
    img1_path = os.path.join(img_path, img1_name)
    label_path = os.path.join(ground_path, subject_name+'.nii.gz')
    
    for msk_item in mask_subjects:
        msk_item_name = msk_item.split('_0000')[0]
        if subject_name == msk_item_name:
            msk_sbj_path = os.path.join(mask_path, msk_item, 'body_BBox.nii.gz')
    
            img_array1, img_itk1, _, img_spacing1, img_origin1, img_direction1 = read_nifti(img0_path)
            img_array2, _, _, _, _, _ = read_nifti(img1_path)
            label_array, _, _, _, _, _ = read_nifti(label_path)
            mask_array, mask_itk, _, _, _, _ = read_nifti(msk_sbj_path)
            array_size = img_array1.shape

            # intensity windowing
            img_array1 = windowing_intensity(img_array1, min_ct_int, max_ct_int)
            # img_array2 = windowing_intensity(img_array2, min_pt_int, max_pt_int)
            # bounding box coordinates
            lsif = itk.LabelShapeStatisticsImageFilter()
            lsif.Execute(mask_itk)
            boundingBox = np.array(lsif.GetBoundingBox(1))
            x_start, y_start, z_start, x_size, y_size, z_size = boundingBox
            
            x_end = x_start+x_size
            y_end = y_start+y_size
            z_end = z_start+z_size
    
            masked_img1 = mask_array*img_array1
            masked_img2 = mask_array*img_array2
            masked_label = mask_array*label_array
    
            cropped_array1 = np.zeros((z_size, y_size, x_size))
            cropped_array1 = masked_img1[z_start:z_end, y_start:y_end, x_start:x_end]
            
            cropped_array2 = np.zeros((z_size, y_size, x_size))
            cropped_array2 = masked_img2[z_start:z_end, y_start:y_end, x_start:x_end]

            cropped_label = np.zeros((z_size, y_size, x_size))
            cropped_label = masked_label[z_start:z_end, y_start:y_end, x_start:x_end]
            
            cropped_path1 = os.path.join(cropped_path_img, img0_name)
            cropped_path2 = os.path.join(cropped_path_img, img1_name)
            cropped_path3 = os.path.join(cropped_path_lablel,  subject_name+'.nii.gz')
    
            write_nifti_from_vol(cropped_array1, img_origin1, img_spacing1, img_direction1, cropped_path1)
            write_nifti_from_vol(cropped_array2, img_origin1, img_spacing1, img_direction1, cropped_path2)
            write_nifti_from_vol(cropped_label, img_origin1, img_spacing1, img_direction1, cropped_path3)
    
            logs = {}
            logs['orig_array_size'] = array_size
            logs['z_start'] = z_start
            logs['z_end'] = z_end
            logs['y_start'] = y_start
            logs['y_end'] = y_end
            logs['x_start'] = x_start
            logs['x_end'] = x_end
            logs['orders'] = 'array[z_start:z_end, y_start:y_end, x_start:x_end]'
            pkl_path = os.path.join(log_path, subject_name+'.pkl')
            write_pickle(pkl_path, logs)
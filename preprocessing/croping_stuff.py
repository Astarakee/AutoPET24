import os
import numpy as np
import SimpleITK as itk
from utils.sitk_stuff import read_nifti
from utils.json_pickle_stuff import write_pickle
from utils.writer import write_nifti_from_itk, write_nifti_from_vol
from utils.paths_dirs_stuff import create_path, path_contents, path_contents_pattern
def get_bounding_box(body_mask_out_path):

    subjects = os.listdir(body_mask_out_path)
    xy_margin = 15
    n_subjet = len(subjects)

    for ixx, subject in enumerate(subjects):
        print('Getting Body Bbox from mask of subject {} our of {}'.format(ixx+1, n_subjet))

        subject_path = os.path.join(body_mask_out_path, subject, 'body.nii.gz')
        write_path = os.path.join(body_mask_out_path, subject, 'body_extremities_BBox.nii.gz')

        _, body_itk, _, body_spacing, body_origin, body_direction = read_nifti(subject_path)

        # largets connect component contains only body
        component_image = itk.ConnectedComponent(body_itk)
        sorted_component_image = itk.RelabelComponent(component_image, sortByObjectSize=True)
        largest_component_binary_image = sorted_component_image == 1
        itk_array = itk.GetArrayFromImage(largest_component_binary_image)

        # bounding box coordinates
        lsif = itk.LabelShapeStatisticsImageFilter()
        lsif.Execute(largest_component_binary_image)
        boundingBox = np.array(lsif.GetBoundingBox(1))
        x_start, y_start, z_start, x_size, y_size, z_size = boundingBox

        z_dim , y_dim, x_dim = itk_array.shape

        x_end = x_start +x_size
        y_end = y_start +y_size
        z_end = z_dim # get all axial slices
        z_start = 0
        # add margins for sanity check
        x_start -= xy_margin
        if x_start <0:
            x_start = 0
        x_end += xy_margin
        if x_end >x_dim:
            x_end = x_dim
        y_start -= xy_margin
        if y_start <0:
            y_start = 0
        y_end += xy_margin
        if y_end >y_dim:
            y_end = y_dim

        # creating masks for orthogonal views
        new_array1 = np.zeros_like(itk_array)
        new_array2 = np.zeros_like(itk_array)
        new_array3 = np.zeros_like(itk_array)
        new_array1[z_start:z_end ,:, :] = 10
        new_array2[:, :, x_start:x_end] = 10
        new_array3[:, y_start:y_end, :] = 10
        temp_new_array = new_array1 +new_array2 +new_array3
        # binarizing the mask
        temp_new_array[temp_new_array==30] = 100
        temp_new_array[temp_new_array!=100] = 0
        temp_new_array[temp_new_array==100] = 1
        temp_new_array = temp_new_array.astype('uint8')

        new_itk = itk.GetImageFromArray(temp_new_array)
        write_nifti_from_itk(new_itk, body_origin, body_spacing, body_direction, write_path)

    return None



def get_cropped_volumes(nifti_in_path, body_mask_out_path, nnunet_in_path, crop_log_path):
    create_path(nnunet_in_path)
    create_path(crop_log_path)
    img_path = os.path.join(nifti_in_path, 'imagesTr')
    mask_path = os.path.join(nifti_in_path, 'labelsTr')
    ch1_files = path_contents_pattern(img_path, '_0000.nii.gz')
    ch2_files = path_contents_pattern(img_path, '_0001.nii.gz')
    mask_files = path_contents_pattern(mask_path, '.nii.gz')
    cropped_folders = path_contents(body_mask_out_path)
    n_case = len(cropped_folders)

    for ixx in range(n_case):
        print('creating cropped image {} out of {}'.format(ixx +1, n_case))

        current_crop = cropped_folders[ixx]
        current_ch1 = ch1_files[ixx]
        current_ch2 = ch2_files[ixx]
        current_mask = mask_files[ixx]

        folder_name = current_crop.split('_0000')[0]
        ch1_file_name = current_ch1.split('_0000.nii')[0]
        ch2_file_name = current_ch2.split('_0001.nii')[0]
        mask_file_name = current_mask.split('.nii.gz')[0]

        if folder_name == ch1_file_name == ch2_file_name == mask_file_name:

            cropped_folder_path = os.path.join(body_mask_out_path, current_crop)
            crop_mask_path = os.path.join(cropped_folder_path, 'body_extremities_BBox.nii.gz')
            ch1_file_path = os.path.join(img_path, current_ch1)
            ch2_file_path = os.path.join(img_path, current_ch2)
            mask_file_path = os.path.join(mask_path, current_mask)

            crop_array, crop_itk, _, crop_spacing, crop_origin, crop_direction = read_nifti(crop_mask_path)
            ch1_array, _, _, _, _, _ = read_nifti(ch1_file_path)
            ch2_array, _, _, _, _, _ = read_nifti(ch2_file_path)
            mask_array, _, _, _, _, _ = read_nifti(mask_file_path)

            array_size = crop_array.shape
            # bounding box coordinates
            lsif = itk.LabelShapeStatisticsImageFilter()
            lsif.Execute(crop_itk)
            boundingBox = np.array(lsif.GetBoundingBox(1))
            x_start, y_start, z_start, x_size, y_size, z_size = boundingBox
            x_end = x_start +x_size
            y_end = y_start +y_size
            z_end = z_start +z_size

            masked_ch1 = crop_array *ch1_array
            masked_ch2 = crop_array *ch2_array
            masked_mask = crop_array *mask_array
            cropped_ch1 = np.zeros((z_size, y_size, x_size))
            cropped_ch2 = np.zeros((z_size, y_size, x_size))
            cropped_mask = np.zeros((z_size, y_size, x_size))
            cropped_ch1 = masked_ch1[z_start:z_end, y_start:y_end, x_start:x_end]
            cropped_ch2 = masked_ch2[z_start:z_end, y_start:y_end, x_start:x_end]
            cropped_mask = masked_mask[z_start:z_end, y_start:y_end, x_start:x_end]

            cropped_path1 = os.path.join(nnunet_in_path, 'imagesTr', current_ch1)
            cropped_path2 = os.path.join(nnunet_in_path, 'imagesTr', current_ch2)
            cropped_path_mask = os.path.join(nnunet_in_path, 'labelsTr', current_mask)
            write_nifti_from_vol(cropped_ch1, crop_origin, crop_spacing, crop_direction, cropped_path1)
            write_nifti_from_vol(cropped_ch2, crop_origin, crop_spacing, crop_direction, cropped_path2)
            write_nifti_from_vol(cropped_mask, crop_origin, crop_spacing, crop_direction, cropped_path_mask)

            logs = {}
            logs['orig_array_size'] = array_size
            logs['z_start'] = z_start
            logs['z_end'] = z_end
            logs['y_start'] = y_start
            logs['y_end'] = y_end
            logs['x_start'] = x_start
            logs['x_end'] = x_end
            logs['orders'] = 'array[z_start:z_end, y_start:y_end, x_start:x_end]'
            pkl_path = os.path.join(crop_log_path, folder_name +'.pkl')
            write_pickle(pkl_path, logs)

    return None

body_mask_path = '/mnt/workspace/data/AutoPET/Croping_Masks_Lymphoma'
fullres_in = '/mnt/workspace/data/AutoPET/Dataset702_AutoPetLymph/'
nnunet_in = '/mnt/workspace/data/AutoPET/AutoPET_Lymphoma_Cropped'
crop_log_path = '/mnt/workspace/data/AutoPET/AutoPET_Lymphoma_Cropped/crop_logs'
get_bounding_box(body_mask_path)
get_cropped_volumes(fullres_in, body_mask_path, nnunet_in, crop_log_path)
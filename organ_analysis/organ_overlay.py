import os
import numpy as np
from tools.paths_dirs_stuff import path_contents_pattern, path_contents
from tools.sitk_stuff import read_nifti
from tools.json_pickle_stuff import write_pickle, read_pickle

gt_path = '/home/mehdi/Data/AutoPET24/Dataset704_AutoPET24Crop/labelsTr'
totalseg_path = '/home/mehdi/Data/AutoPET24/Dataset704_AutoPET24Crop/total_seg'
save_path = '/home/mehdi/Data/AutoPET24/'
log_name = 'totalseg_overlay_organs'
pkl_save_path = os.path.join(save_path, log_name+'.pkl')


totalseg_subjects = path_contents(totalseg_path)
gt_subjects = path_contents_pattern(gt_path, '.nii.gz')

logs = {}

pos_heart = 0
pos_left_kidney = 0
pos_right_kidney = 0
pos_brain = 0
for ix, item in enumerate(gt_subjects):
    print('-'*10)
    print('subject {} out of {}'.format(ix+1, len(gt_subjects)))
    item_name = item.split('.nii.gz')[0]
    gt_abs_path = os.path.join(gt_path, item)
    totalseg_abs_path = os.path.join(totalseg_path, item_name)
    totalseg_heart = path_contents_pattern(totalseg_abs_path, 'heart.nii.gz')[0]
    totalseg_kidney_l = path_contents_pattern(totalseg_abs_path, 'kidney_left.nii.gz')[0]
    totakseg_kindnet_r = path_contents_pattern(totalseg_abs_path, 'kidney_right.nii.gz')[0]
    totalseg_brain = path_contents_pattern(totalseg_abs_path, 'brain.nii.gz')[0]
    heart_abs_path = os.path.join(totalseg_abs_path, totalseg_heart)
    l_kidney_abs_path = os.path.join(totalseg_abs_path, totalseg_kidney_l)
    r_kidney_abs_path = os.path.join(totalseg_abs_path, totakseg_kindnet_r)
    brain_abs_path = os.path.join(totalseg_abs_path, totalseg_brain)

    sg_array, sg_itk, _, _, _, _ = read_nifti(gt_abs_path)
    sg_labels = np.unique(sg_array)
    if 1 in sg_labels: # if positive case
        temp = {}
        heart_array, _, _, _, _, _ = read_nifti(heart_abs_path)
        l_kidney_array, _, _, _, _, _ = read_nifti(l_kidney_abs_path)
        r_kidney_array, _, _, _, _, _ = read_nifti(r_kidney_abs_path)
        brain_array, _, _, _, _, _ = read_nifti(brain_abs_path)

        heart_overlay = sg_array*heart_array
        l_kidney_overlay = sg_array*l_kidney_array
        r_kidney_overlay = sg_array*r_kidney_array
        brain_overlay = sg_array*brain_array

        heart_overlay_labels = np.unique(heart_overlay)
        l_kidney_overlay_labels = np.unique(l_kidney_overlay)
        r_kidney_overlay_labels = np.unique(r_kidney_overlay)
        brain_overlay_labels = np.unique(brain_overlay)

        if 1 in heart_overlay_labels:
            n_heart = len(np.where(heart_overlay==1)[0])
            pos_heart += 1
            temp['n_heart'] = n_heart
            print('positive heart cases are:', pos_heart)

        elif 1 in l_kidney_overlay_labels:
            n_lk = len(np.where(l_kidney_overlay==1)[0])
            pos_left_kidney += 1
            temp['n_left_kidney'] = n_lk
            print('positive left kindey cases are:', pos_left_kidney)

        elif 1 in r_kidney_overlay_labels:
            n_rk = len(np.where(r_kidney_overlay==1)[0])
            pos_right_kidney += 1
            temp['n_right_kidney'] = n_rk
            print('positive right kindey cases are:', pos_right_kidney)

        elif 1 in brain_overlay_labels:
            n_brain = len(np.where(brain_overlay==1)[0])
            pos_brain += 1
            temp['n_brain'] = n_brain
            print('positive brain cases are:', pos_brain)

        logs[item] = temp


write_pickle(pkl_save_path, logs)
mylogs = read_pickle(pkl_save_path)
import os
import SimpleITK as itk
import numpy as np
from scipy.ndimage import morphology
import os



pred_path = "/mnt/workspace/data/nnUnet/nnUNet_results/AutoPET24/Dataset704_AutoPET24Crop/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation"
save_path = "/home/mehdi/Data/AutoPET24/postprocessing/DilateErod5/fold_0"
pred_files = os.listdir(pred_path)
pred_files = [x for x in pred_files if '.nii.gz' in x]
n_subject = len(pred_files)

structure=np.ones((5,5,5))

log_summary = []
for ix, case in enumerate(pred_files):
    print('\n'*5)
    print("working on subject {} out of {}".format(ix+1, n_subject))
    pred_abs_path = os.path.join(pred_path, case)
    itk_obj = itk.ReadImage(pred_abs_path)
    mask_array = itk.GetArrayFromImage(itk_obj)
    mask_array_dilate = morphology.binary_dilation(mask_array, structure=structure)
    mask_array_erod = morphology.binary_erosion(mask_array_dilate, structure=structure)
    mask_array_new = mask_array_erod.astype(np.uint8)
    new_itk = itk.GetImageFromArray(mask_array_new)
    new_itk.SetOrigin(itk_obj.GetOrigin())
    new_itk.SetSpacing(itk_obj.GetSpacing())
    new_itk.SetDirection(itk_obj.GetDirection())
    save_abs_path = os.path.join(save_path, case)
    itk.WriteImage(new_itk, save_abs_path)

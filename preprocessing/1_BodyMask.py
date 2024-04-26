import os
from tools.paths_dirs_stuff import path_contents_pattern, create_path


def get_body_mask(nifti_in_path, body_mask_out_path):
    create_path(body_mask_out_path)
    full_res_files = path_contents_pattern(nifti_in_path, '_0000.nii.gz')
    
    failed_subjects = []
    for file in full_res_files:
        file_name = file.split('.nii.gz')[0]
        src_file = os.path.join(nifti_in_path, file)
        dst_folder = os.path.join(body_mask_out_path, file_name)
        
        try:
            os.system('TotalSegmentator -i %s -o %s -ta body --fast' % (src_file, dst_folder))
            seg_files = os.listdir(dst_folder)
            target_file = 'body.nii.gz'
            seg_files.remove(target_file)
            for seg in seg_files:
                file_path = os.path.join(dst_folder, seg)
                os.remove(file_path)
        except Exception:
            failed_subjects.append(file) # can be saved as logs.
            
    return None


nifti_path_in = '/home/mehdi/Data/AutoPET/imagesTr'
crop_path = '/home/mehdi/Data/Crop_AutoPET/CropMask'

get_body_mask(nifti_path_in, crop_path)

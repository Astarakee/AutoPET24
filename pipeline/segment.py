import os
from tools.paths_dirs_stuff import create_path
import time

out_path = '/home/mehdi/Data/AutoPET24/Testing_preds'
nnunet_in_path = os.path.join(out_path, 'imagesTr')
pred_crop_out = os.path.join(out_path, 'pred_crops')

os.system('nnUNetv2_predict -i %s -o %s -d 704 -c 3d_fullres -p nnUNetResEncUNetLPlans' % (nnunet_in_path, pred_crop_out))


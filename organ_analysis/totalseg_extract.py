import os
from tools.paths_dirs_stuff import path_contents_pattern, create_path

data_path = '/home/mehdi/Data/AutoPET24/Dataset704_AutoPET24Crop/imagesTr'
save_path = '/home/mehdi/Data/AutoPET24/Dataset704_AutoPET24Crop/total_seg'
ct_files = path_contents_pattern(data_path, '_0000.nii.gz')
n_files = len(ct_files)

for ix, item in enumerate(ct_files):
    print('-'*6)
    print('subject {} out of {} in progress...'.format(ix+1, n_files))
    item_name = item.split('_0000')[0]
    in_data = os.path.join(data_path, item)
    out_data = os.path.join(save_path, item_name)
    create_path(out_data)
    os.system('TotalSegmentator -i %s -o %s --fast' % (in_data, out_data))


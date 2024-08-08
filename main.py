import os
import time
import argparse
from tools.preprocess import run_prepare
from tools.reconstruction import run_fullres
from tools.paths_dirs_stuff import create_path


parser = argparse.ArgumentParser(description='AutoPET2024 Challenge')
parser.add_argument('-i', type=str, help='abs path to directory containing nifti in decathlon format', required=True)
parser.add_argument('-o', type=str, help='abs path to save results', required=True)
args = parser.parse_args()
in_path = args.i
out_path = args.o
# in_path = '/home/mehdi/Data/AutoPET24/TestingPhase'
# out_path = '/home/mehdi/Data/AutoPET24/Testing_preds'
def main():
    start_time = time.time()
    nnunet_in_path = os.path.join(out_path, 'imagesTr')
    pred_crop_out = os.path.join(out_path, 'pred_crops')
    create_path(pred_crop_out)
    print('-'*5, 'preprocessing data')
    run_prepare(in_path, out_path)
    print('-' * 5, 'nnunet prediction')
    os.system('nnUNetv2_predict -i %s -o %s -d 704 -c 3d_fullres -p nnUNetResEncUNetLPlans' % (nnunet_in_path, pred_crop_out))
    print('-' * 5, 'postprocessing predicted masks')
    run_fullres(out_path)
    print('\n'*5, 'prediction completed in {:.4f} seconds'.format((time.time() - start_time)))
    return None

if __name__ == '__main__':
    main()
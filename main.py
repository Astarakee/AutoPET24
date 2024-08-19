import os
import time
import argparse
from tools.preprocess import run_prepare
from tools.paths_dirs_stuff import create_path
from tools.reconstruction import run_fullres, save_datacentric

parser = argparse.ArgumentParser(description='AutoPET2024 Challenge')
parser.add_argument('-i', type=str, help='abs path to input directory', required=True)
parser.add_argument('-o', type=str, help='abs path to output directory', required=True)
args = parser.parse_args()
in_path = args.i
out_path = args.o
def main():
    start_time = time.time()
    path_prepared_data = os.path.join(out_path, 'temp_data')
    nnunet_path_out = os.path.join(path_prepared_data, "nnunet_out")
    create_path(path_prepared_data)
    create_path(nnunet_path_out)
    run_prepare(in_path, path_prepared_data)
    nnunet_path_in = os.path.join(path_prepared_data, 'imagesTs')
    print('-' * 5, 'nnunet prediction')
    os.system('nnUNetv2_predict -i %s -o %s -d 704 -c 3d_fullres -p nnUNetResEncUNetLPlans -f 0 1 2' % (nnunet_path_in, nnunet_path_out))
    print('-' * 5, 'postprocessing predicted masks')
    run_fullres(out_path, path_prepared_data, nnunet_path_out)
    save_datacentric(out_path, False)
    print('\n'*5, 'prediction completed in {:.4f} seconds'.format((time.time() - start_time)))
    return None

if __name__ == '__main__':
    main()


# in_path = '/mnt/workspace/data/00_junks/AutoPET24/input'
# out_path = '/mnt/workspace/data/00_junks/AutoPET24/output'
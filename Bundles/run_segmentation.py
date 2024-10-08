import pandas as pd
import subprocess
import torch
from   nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from pathlib import Path
import os

def perform_Tracer_aware_Segmentation(csv_file,model_mapping, model_folders,output_folder):

    step_size = 0.5
    disable_tta = False
    verbose = None
    disable_progress_bar = False

    chk = 'checkpoint_final.pth'
    save_probabilities = None
    continue_prediction = None
    npp = 3
    nps = 3
    prev_stage_predictions = None

    inputs = {}
    for val in list(model_mapping.values()):
        inputs[val] = []


    df = pd.read_csv(csv_file,header=None,names=["Filename","Class"])

    for row in df.iterrows():
        matching_files = ['/input/'+filename for filename in os.listdir('/input') if filename.startswith(Path(row[1].Filename).name[:-len("_0000.nii.gz")])]

        matching_files.sort()
        inputs[model_mapping[(row[1].Class)]].append(matching_files)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda')

    for val in list(model_mapping.values()):



        predictor = nnUNetPredictor(tile_step_size=step_size,
                                    use_gaussian=True,
                                    use_mirroring=not disable_tta,
                                    perform_everything_on_device=True,
                                    device=device,
                                    verbose=verbose,
                                    allow_tqdm=not disable_progress_bar,
                                    verbose_preprocessing=verbose)

        predictor.initialize_from_trained_model_folder(model_folders[val], [0,1], chk)
        predictor.predict_from_files(inputs[val], output_folder, save_probabilities=save_probabilities,
                                     overwrite=not continue_prediction,
                                     num_processes_preprocessing=npp,
                                    num_processes_segmentation_export=nps,
                                    folder_with_segs_from_prev_stage=prev_stage_predictions,
                                    num_parts=1, part_id=0)



if __name__ == "__main__":
    model_mapping = {
        0: "FDG",
        1: "PSMA"
    }
    model_folders = {
        "FDG": "/opt/Dataset704_AutoPET24CropFDG/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
        "PSMA": "/opt/Dataset704_AutoPET24CroPSMA/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
    }
    output_folder = "/output"
    perform_Tracer_aware_Segmentation('/output/PET_Tracer_predictions.csv',model_mapping, model_folders,output_folder)
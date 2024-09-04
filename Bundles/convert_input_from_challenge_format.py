import SimpleITK as sitk
from PyMAIA.utils.file_utils import subfiles
import click
from pathlib import Path
from tools.preprocess import run_prepare
from tools.paths_dirs_stuff import create_path
from tools.reconstruction import run_fullres, save_datacentric

@click.group()
def cli1():
    # create group for all the commands so you can
    # run them from the __name__ == "__main__" block
    pass


@click.group()
def cli2():
    # create group for all the commands so you can
    # run them from the __name__ == "__main__" block
    pass

@click.group()
def cli3():
    # create group for all the commands so you can
    # run them from the __name__ == "__main__" block
    pass

@click.group()
def cli4():
    # create group for all the commands so you can
    # run them from the __name__ == "__main__" block
    pass


@cli1.command()
@click.option("--input-mha-folder", type=str, required=True)
@click.option("--input-nifti-folder", type=str, required=True)
def convert_input_from_challenge_format(input_mha_folder, input_nifti_folder):

    Path(input_nifti_folder).mkdir(exist_ok=True, parents=True)
    ct_files = subfiles(str(Path(input_mha_folder).joinpath("images","ct")), suffix=".mha",join=False)
    for ct_file in ct_files:
        output_file = ct_file[:-4] + "_0000.nii.gz"
        img = sitk.ReadImage(str(Path(input_mha_folder).joinpath("images","ct",ct_file)))
        sitk.WriteImage(img, str(Path(input_nifti_folder).joinpath(output_file)))
    pet_files = subfiles(str(Path(input_mha_folder).joinpath("images","pet")), suffix=".mha",join=False)
    for pet_file in pet_files:
        output_file = pet_file[:-4] + "_0001.nii.gz"
        img = sitk.ReadImage(str(Path(input_mha_folder).joinpath("images","pet",pet_file)))
        sitk.WriteImage(img, str(Path(input_nifti_folder).joinpath(output_file)))

@cli2.command()
@click.option("--output-mha-folder", type=str, required=True)
@click.option("--output-nifti-folder", type=str, required=True)
def convert_output_to_challenge_format(output_mha_folder, output_nifti_folder):

    Path(output_mha_folder).joinpath("images","automated-petct-lesion-segmentation").mkdir(exist_ok=True, parents=True)
    pred_files = subfiles(str(Path(output_nifti_folder)), suffix=".nii.gz",join=False)
    for pred_file in pred_files:
        output_file = pred_file[:-7] + ".mha"
        img = sitk.ReadImage(str(Path(output_nifti_folder).joinpath(pred_file)))
        sitk.WriteImage(img, str(Path(output_mha_folder).joinpath("images","automated-petct-lesion-segmentation",output_file)))


@cli3.command()
@click.option("--input-folder", type=str, required=True)
@click.option("--preprocess-folder", type=str, required=True)
def preprocess_images(input_folder, preprocess_folder):


    Path(preprocess_folder).mkdir(exist_ok=True, parents=True)

    run_prepare(input_folder, preprocess_folder)
    #nnunet_path_in = os.path.join(preprocess_folder, 'imagesTs')

@cli4.command()
@click.option("--output-folder", type=str, required=True)
@click.option("--output-nifti-folder", type=str, required=True)
@click.option("--preprocess-folder", type=str, required=True)
def run_postprocessing(output_folder, preprocess_folder, output_nifti_folder):
    run_fullres(output_folder, preprocess_folder, output_nifti_folder)
    save_datacentric(output_folder, False)

cli = click.CommandCollection(sources=[cli1, cli2, cli3, cli4])


if __name__ == "__main__":
    cli()
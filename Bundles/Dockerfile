FROM continuumio/miniconda3

RUN conda create -n MONAI python=3.10

RUN conda install -n MONAI pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
RUN /opt/conda/envs/MONAI/bin/pip install nnunetv2 "monai[nibabel, skimage, scipy, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, pandas, mlflow, matplotlib, pydicom]" fire

RUN apt-get update && apt-get install zip unzip -y


# Links expiring on 2024-08-24
ENV CLASSIFIER_LINK="https://minio.admin.app.cloud.cbh.kth.se/autopet24/AutoPET_PET_Classification.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=maia-admin%2F20240817%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240817T053828Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=08b1cd686a16866f3d4bd3aab58db5507542c89fd3e8967c5c0b1b1a4eb4291d"
ENV FDG_LINK="https://minio.admin.app.cloud.cbh.kth.se/autopet24/Dataset704_AutoPET24CropFDG.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=maia-admin%2F20240817%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240817T053904Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=715ae1a14bbd159da60cd6857b8d78e4931bf97efb944a3f2655da36b91c95fe"
ENV PSMA_LINK="https://minio.admin.app.cloud.cbh.kth.se/autopet24/Dataset704_AutoPET24CroPSMA.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=maia-admin%2F20240817%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240817T053542Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=6b5856a47e1449a8f910a57eecd0142d0b7a62405ecd9744e54054afe3d4717a"

RUN wget  ${CLASSIFIER_LINK} -O /opt/AutoPET_PET_Classification.zip
RUN unzip /opt/AutoPET_PET_Classification.zip -d /opt

RUN wget ${FDG_LINK} -O /opt/Dataset704_AutoPET24CropFDG.zip
RUN unzip /opt/Dataset704_AutoPET24CropFDG.zip -d /opt


RUN wget ${PSMA_LINK}  -O /opt/Dataset704_AutoPET24CropPSMA.zip
RUN unzip /opt/Dataset704_AutoPET24CropPSMA.zip -d /opt


COPY inference.sh /opt/

COPY run_segmentation.py /opt/

CMD ["bash", "/opt/inference.sh"]

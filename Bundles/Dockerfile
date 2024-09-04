FROM continuumio/miniconda3

RUN conda create -n MONAI python=3.10

RUN conda install -n MONAI pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
RUN /opt/conda/envs/MONAI/bin/pip install pymaia-learn nnunetv2 "monai[nibabel, skimage, scipy, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, pandas, mlflow, matplotlib, pydicom]" fire

RUN apt-get update && apt-get install zip unzip -y


# Links expiring on 2024-08-24
ENV CLASSIFIER_LINK="https://minio.admin.app.cloud.cbh.kth.se/autopet24/AutoPET_PET_Classification.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=maia-admin%2F20240902%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240902T121343Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=bc16b2f9501db274af38a25fb7222f203c098cfac816172d340c613434097f7c"
ENV FDG_LINK="https://minio.admin.app.cloud.cbh.kth.se/autopet24/Dataset704_AutoPET24CropFDG.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=maia-admin%2F20240902%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240902T121231Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=e544960022a315118f39584b6b851a45396165a8c40db1b099714e345c2e885f"
ENV PSMA_LINK="https://minio.admin.app.cloud.cbh.kth.se/autopet24/Dataset704_AutoPET24CroPSMA.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=maia-admin%2F20240902%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240902T121317Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0e0a5dc59db4a96e91fcfe1a3244695518e5d9ceaa24325c6e6476cb96276d99"


RUN wget  ${CLASSIFIER_LINK} -O /opt/AutoPET_PET_Classification.zip
RUN unzip /opt/AutoPET_PET_Classification.zip -d /opt

RUN wget ${FDG_LINK} -O /opt/Dataset704_AutoPET24CropFDG.zip
RUN unzip /opt/Dataset704_AutoPET24CropFDG.zip -d /opt


RUN wget ${PSMA_LINK}  -O /opt/Dataset704_AutoPET24CropPSMA.zip
RUN unzip /opt/Dataset704_AutoPET24CropPSMA.zip -d /opt


COPY inference.sh /opt/

COPY run_segmentation.py /opt/
COPY convert_input_from_challenge_format.py /opt/
COPY tools /opt/tools

RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/tmp/classification_output /input /output/images/automated-petct-lesion-segmentation  && \
    chown -R algorithm:algorithm /opt/tmp/ /input /output

USER algorithm

CMD ["bash", "/opt/inference.sh"]

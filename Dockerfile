FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
LABEL authors="MehdiAstaraki"

RUN apt-get update -y
RUN mkdir -p  /autopet/nnUNet_results
RUN chmod -R 777 /autopet
RUN pip  install nnunetv2
RUN adduser user
USER user

WORKDIR /autopet
COPY main.py .
COPY tools tools/
COPY Dataset704_AutoPET24Crop ./nnUNet_results/Dataset704_AutoPET24Crop
ENV nnUNet_results=/autopet/nnUNet_results
CMD ["python", "main.py", "-i", "/input", "-o", "/output"]

#docker build . -t autopet:v1
#docker run --rm --gpus all --network none --memory="30g" -v /mnt/workspace/data/00_junks/AutoPET24/input:/input -v /mnt/workspace/data/00_junks/AutoPET24/output:/output --shm-size 4g autopet:v1
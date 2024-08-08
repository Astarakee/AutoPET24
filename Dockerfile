FROM  nvcr.io/nvidia/pytorch:23.10-py3
LABEL authors="MehdiAstaraki"

RUN apt-get update -y
RUN mkdir -p  /autopet/nnUNet_results
RUN chmod -R 777 /autopet
RUN pip  install nnunetv2
RUN pip install gdown
RUN adduser user
USER user
WORKDIR /autopet
COPY . /autopet
## checkpoint download
RUN python dl_checkpoint.py
RUN chmod -R 777 /autopet/Dataset704_AutoPET24Crop
RUN mv /autopet/Dataset704_AutoPET24Crop /autopet/nnUNet_results
ENV nnUNet_results=/autopet/nnUNet_results
CMD ["python", "main.py", "-i", "/input", "-o", "/output"]


#docker build . -t autopet:v1
#docker run --rm --gpus all --network none --memory="32g" -v <in>:/input/images/ -v <out>:/output/images/ --shm-size 4g autopet:v1
# Tracer-aware PET-CT Segmentation

To create the Docker image for the Tracer-aware PET-CT Segmentation Bundle, run the following command:

```bash
docker build -t autopet:v1-tracer_classification -f Dockerfile .
```

And to run the Docker container, use the following command:

```bash
docker run --rm --gpus all --network none --memory="32g" -v <in>:/input/ -v <out>:/output/ --shm-size 4g autopet:v1-tracer_classification
```
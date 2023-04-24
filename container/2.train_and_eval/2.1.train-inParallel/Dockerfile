# ARG CUDA_VER=11.8
# ARG LINUX_VER=ubuntu20.04
# ARG PYTHON_VER=3.10
# ARG RAPIDS_VER=23.06
# ARG FROM_IMAGE=gpuci/rapidsai

# FROM ${FROM_IMAGE}:${RAPIDS_VER}-cuda${CUDA_VER}-runtime-${LINUX_VER}-py${PYTHON_VER}
FROM python:3.8
RUN mkdir /code　
COPY requirements.txt /code/
COPY src/main.py /code/

WORKDIR /code
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "./main.py"]
CMD ["0"]

## For GPU
# FROM nvidia/cuda:12.1.0-base-ubuntu18.04
# #FROM gcr.io/deeplearning-platform-release/tf-gpu

# RUN apt-get update
# RUN apt-get install -y --no-install-recommends python3-pip
# RUN pip3 install --upgrade pip

# RUN mkdir /code　
# COPY requirements.txt /code/
# COPY src/main.py /code/

# WORKDIR /code
# RUN pip3 install -r requirements.txt

# ENTRYPOINT ["python3", "/code/main.py"]
# CMD ["0"]
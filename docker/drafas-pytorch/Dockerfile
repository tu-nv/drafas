from pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y vim
RUN pip install fastapi uvicorn python-multipart

COPY rootfs/ /

WORKDIR /

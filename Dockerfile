# syntax=docker/dockerfile:1

FROM nvidia/cuda:10.0-base-ubuntu18.04

ENV LANG C.UTF-8 

RUN apt update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.8 -y
RUN apt install python3-pip -y
RUN pip3 install --upgrade pip

RUN mkdir workspace

COPY . /workspace

WORKDIR /workspace

RUN pip3 install paddlepaddle

RUN pip3 install -r requirements.txt

CMD [ "python3", "generate_results.py"]

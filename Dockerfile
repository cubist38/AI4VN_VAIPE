# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir workspace

COPY . /workspace

WORKDIR /workspace

RUN pip3 install -r requirements.txt

CMD [ "python3", "generate_results.py"]
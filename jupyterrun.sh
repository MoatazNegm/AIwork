#!/bin/sh
export HF_AUTH_TOKEN="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"
whoami > /root/who
docker rm -f torchjup 2>/dev/null
docker rm -f redis
#docker run --gpus all -d --rm --shm-size 10G -v /moataz-work:/workspace/moataz-work --name torchjup -p 8888:8888 nvcr.io/nvidia/pytorch:24.04-py3 jupyter-notebook --NotebookApp.token='tmatem'
docker run --gpus all -d --rm --memory 10g --memory-swap 40g --shm-size 10G -v /moataz-work:/workspace/moataz-work --name torchjup -p 8888:8888 -p 9000:9000 torchadvrag jupyter-notebook --NotebookApp.token='tmatem'
docker run -d --rm --name redis -p 6379:6379 redis
docker run --rm --net=host -it -d --name therok -e NGROK_AUTHTOKEN=2x2Tulw2EmfN2bbNUs2XUwXg7bI_4keZpBcPUUkXV28ZKGFG ngrok/ngrok:latest http --url=seriously-rested-pup.ngrok-free.app 8888


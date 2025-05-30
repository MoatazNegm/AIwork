#!/bin/sh
export HF_AUTH_TOKEN="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"
whoami > /root/who
docker rm -f torchjup 2>/dev/null
docker rm -f redis
docker rm -f wetty
docker rm flask
docker rm therok
docker run -d --rm --memory 4g  --memory-swap 40g --shm-size 10G -v /moataz-work:/workspace/moataz-work ubuntu
#docker run --gpus all -d --rm --shm-size 10G -v /moataz-work:/workspace/moataz-work --name torchjup -p 8888:8888 nvcr.io/nvidia/pytorch:24.04-py3 jupyter-notebook --NotebookApp.token='tmatem'
docker run --gpus all -d --rm --memory 10g --memory-swap 40g --shm-size 10G -v /moataz-work:/workspace/moataz-work --name torchjup -p 8888:8888 -p v9000:9000 torchadvrag jupyter-notebook --NotebookApp.token='tmatem'
docker run -d --rm --name redis -p 6379:6379 redis
docker run --rm --net=host -it -d --name therok -e NGROK_AUTHTOKEN=2x2Tulw2EmfN2bbNUs2XUwXg7bI_4keZpBcPUUkXV28ZKGFG ngrok/ngrok:latest http --url=seriously-rested-pup.ngrok-free.app 8888
docker run --rm -p 4000:80 -v /moataz-work/flask/src/:/app --name flask -d flask-adminlte-app
docker run -d --rm --name wetty -p 3000:3000 moataznegm/quickstor:wetty --ssh-host=192.168.8.10 --ssh-user=root --base=/ moataznegm/qu
ickstor:wetty
docker run --rm -p 4041:4040 -d -it  --name thewetrok -e NGROK_AUTHTOKEN=2xiFeIAkqBLENfcmgwPIcwIxiwI_3ivqeaG3WgLW2VfjoSapZ ngrok/ngrok:latest http http://192.168.8.10:3000 --url=artistic-merely-lynx.ngrok-free.app
docker run --rm -p 4042:4040 -d -it  --name therok2 -e NGROK_AUTHTOKEN=2xiZBpkVYQH7cRpRv9gfQAwazCE_3mYF3X8wq9U4xWptFyP95 ngrok/ngrok:latest http http://192.168.8.10:4000 --url=driven-shrimp-nicely.ngrok-free.app

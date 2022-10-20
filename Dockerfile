FROM tensorflow/tensorflow:latest-gpu

RUN apt update -y
RUN apt install x11-apps -y

RUN pip3 install matplotlib nptyping

FROM tensorflow/tensorflow:latest-gpu
RUN apt install -y python3-tk
RUN pip3 install matplotlib nptyping

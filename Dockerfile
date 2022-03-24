FROM gcr.io/tpu-pytorch/xla:r1.10_3.8_tpuvm

RUN mkdir /vits-robustness-torch/
WORKDIR /vits-robustness-torch/
COPY . .
RUN sudo pip3 install -r requirements.txt
RUN rm -r datasets notebooks sweeps
FROM gcr.io/tpu-pytorch/xla:r1.10_3.8_tpuvm

RUN sudo pip3 uninstall tensorflow --yes
RUN sudo pip3 install -r requirements.txt
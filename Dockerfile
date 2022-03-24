FROM gcr.io/tpu-pytorch/xla:r1.10_3.8_tpuvm

RUN pip install --upgrade pip

RUN mkdir /vits-robustness-torch/
WORKDIR /vits-robustness-torch/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install wandb

COPY . .
RUN rm -r datasets notebooks
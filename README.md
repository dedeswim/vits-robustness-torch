# ViTs Robustness

# Pre-requisites

This repo works with:

- Python `3.8.10`, and will probably work with newer versions.
- `torch==1.8.1` and `1.10.1`, and it will probably work with PyTorch `1.9.x`.
- `torchvision==0.9.1` and `0.11.2`, and it will probably work with torchvision `0.10.x`.
- The other requirements are in `requirements.txt`, hence they can be installed with `pip install -r requirements.txt`

In case you want to use Weights and Biases, after installing the requisites, install `wandb` with `pip install wandb`, and run `wandb login`.

In case you want to read or write your results to a Google Cloud Storage bucket (which is supported by this repo), install the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud), and login. Then you are ready to use GCS for both storing data and experiments results, as well as download checkpoints by using paths in the form of `gs://bucket-name/path-to-dir-or-file`.

## Training

```bash
DATA_DIR=gs://vits-robustness/tensorflow_datasets \
DATASET=tfds/imagenet2012 \
MODEL=xcit_nano_12_p16_224 \
EXPERIMENT=xcit-nano-adv-training-imagenet \
OUTPUT=gs://vits-robustness/output/train \
CONFIG=configs/xcit-nano-adv-training.yaml \
python launch_xla.py --num-devices 8 train.py $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --output $OUTPUT --log-wandb --model xcit_nano_12_p16_224 --config $CONFIG --epochs 10 --epochs 100 --batch-size 128 --adv-training pgd --no-normalize --attack-steps 1 --log-interval 50 --eps-schedule linear --eps-schedule-period 10
```

## Validation

```bash
DATA_DIR=gs://vits-robustness/tensorflow_datasets \
DATASET=tfds/imagenet2012 \
MODEL=xcit_nano_12_p16_224 \
CHECKPOINT_DIR=gs://vits-robustness/output/train/xcit-nano-adv-training-imagenet-8 \
python launch_xla.py --num-devices 1 validate.py $DATA_DIR --dataset $DATASET --model $MODEL --batch-size 1024 --no-normalize --checkpoint $CHECKPOINT_DIR/last.pth.tar --attack-eps 8
```

In order to run the tests, install pytest via `pip install pytest`, and run

```bash
python -m pytest .
```
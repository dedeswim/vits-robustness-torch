# ViTs Robustness

## Training

```bash
DATA_DIR=gs://vits-robustness/tensorflow_datasets \
DATASET=tfds/imagenet2012 \
MODEL=xcit_nano_12_p16_224 \
EXPERIMENT=xcit-nano-adv-training-imagenet \
OUTPUT=gs://vits-robustness/output/train \
CONFIG=configs/xcit-nano-adv-training.yaml \
python launch_xla.py --num-devices 8 train.py $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --output $OUTPUT --log-wandb --model xcit_nano_12_p16_224 --config $CONFIG --epochs 10 --epochs 100 --batch-size 128 --adv-training pgd --no-normalize --attack-steps 1 --log-interval 50 --attack-lr 1.0 --eps-schedule sine --eps-schedule-period 10
```

## Validation

```bash
DATA_DIR=gs://vits-robustness/tensorflow_datasets \
DATASET=tfds/imagenet2012 \
MODEL=xcit_nano_12_p16_224 \
CHECKPOINT_DIR=gs://vits-robustness/output/train/xcit-nano-adv-training-imagenet-8 \
python launch_xla.py --num-devices 1 validate.py $DATA_DIR --dataset $DATASET --model $MODEL --batch-size 1024 --no-normalize --checkpoint $CHECKPOINT_DIR/last.pth.tar --attack-eps 0.03137254902 --attack-lr 1.0
```
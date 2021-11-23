# ViTs Robustness

## Training

```bash
python launch_xla.py --num-devices 8 train.py gs://vits-robustness/tensorflow_datasets/ --dataset tfds/imagenet2012 --experiment xcit-nano-adv-training-imagenet --output gs://vits-robustness/output/train --log-wandb --model xcit_nano_12_p16_224 --config configs/xcit-nano-adv-training.yaml --epochs 10 --epochs 100 --batch-size 128 --adv-training pgd --no-normalize --attack-steps 1 --log-interval 50 --attack-lr 1.0 --eps-schedule sine --eps-schedule-period 10
```

## Validation

```bash
python launch_xla.py --num-devices 1 validate.py gs://vits-robustness/tensorflow_datasets --dataset tfds/imagenet2012 --model xcit_nano_12_p16_224 --batch-size 1024 --no-normalize --checkpoint gs://vits-robustness/output/train/xcit-nano-adv-training-imagenet-8/last.pth.tar --attack-eps 0.03137254902 --attack-lr 1.0
```
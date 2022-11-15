# A *Light* Recipe to Train Robust Vision Transformers

*Edoardo Debenedetti (ETH Zürich\*), Vikash Sehwag (Princeton University) and Prateek Mittal (Princeton University)*

\*work done as a Master's student at EPFL

Repository for the paper **A *Light* Recipe to Train Robust Vision Transformers**. It contains both the code and the checkpoints of 22 models from the paper relative to 5 different datasets. The codebase is also capable of training models if they are part of, or can be integrated in `timm`, on both GPUs and TPUs. It also supports any 3-channels image classification dataset which is either part of `tfds` or `torchvision`.

## News

The paper was accepted at the first IEEE Conference on Secure and Trustworthy Machine Learning [SaTML](https://satml.org)!

## Abstract

In this paper, we ask whether Vision Transformers (ViTs) can serve as an underlying architecture for improving the adversarial robustness of machine learning models against evasion attacks. While earlier works have focused on improving Convolutional Neural Networks, we show that also ViTs are highly suitable for adversarial training to achieve competitive performance. We achieve this objective using a custom adversarial training recipe, discovered using rigorous ablation studies on a subset of the ImageNet dataset. The canonical training recipe for ViTs recommends strong data augmentation, in part to compensate for the lack of vision inductive bias of attention modules, when compared to convolutions. We show that this recipe achieves suboptimal performance when used for adversarial training. In contrast, we find that omitting all heavy data augmentation, and adding some additional bag-of-tricks ( $\varepsilon$-warmup and larger weight decay), significantly boosts the performance of robust ViTs. We show that our recipe generalizes to different classes of ViT architectures and large-scale models on full ImageNet-1k. Additionally, investigating the reasons for the robustness of our models, we show that it is easier to generate strong attacks during training when using our recipe and that this leads to better robustness at test time. Finally, we further study one consequence of adversarial training by proposing a way to quantify the semantic nature of adversarial perturbations and highlight its correlation with the robustness of the model. Overall, we recommend that the community should avoid translating the canonical training recipes in ViTs to robust training and rethink common training choices in the context of adversarial training.

## Pre-requisites

A large part of the codebase is based on [`timm`](https://github.com/rwightman/pytorch-image-models).

This repo works with:

- Python `3.8.10`, and will probably work with newer versions.
- `torch==1.8.1` and `1.10.1`, and it will probably work with PyTorch `1.9.x` and newer versions.
- `torchvision==0.9.1` and `0.11.2`, and it will probably work with torchvision `0.10.x`.
- The other requirements are in `requirements.txt`, hence they can be installed with `pip install -r requirements.txt`

In case you want to use Weights and Biases, after installing the requisites, install `wandb` with `pip install wandb`, and run `wandb login`.

In case you want to read or write your results to a Google Cloud Storage bucket (which is supported by this repo), install the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud), and login. Then you are ready to use GCS for both storing data and experiments results, as well as download checkpoints by using paths in the form of `gs://bucket-name/path-to-dir-or-file`.

## Training

All these commands are meant to be run on TPU VMs with 8 TPU cores. They can be easily adapted to work on GPUs by using `torch.distributed.launch` (and by removing the `launch_xla.py --num-devices 8` part). The LR can be scaled as explained in the appendix of our paper (which follows DeiT's convention). More info about how to run the training script on TPUs and GPUs can be found in `timm.bits`'s [README](https://github.com/rwightman/pytorch-image-models/tree/bits_and_tpu/timm/bits#timm-bits).

It's possible to train any model which is already part of `timm`. In case a model is not part of `timm`, can be registered to `timm`'s model store using the `@register_model` decorator on a function which instantiates the model. You can take a look at how this can be done [here](https://github.com/dedeswim/vits-robustness-torch/blob/master/src/models/wide_resnet.py#L147).

Moreover, it's possible to use arbitrary loss functions, as long as they have the same structure as [`AdvTrainingLoss`](https://github.com/dedeswim/vits-robustness-torch/blob/master/src/attacks.py#L229) and [`TRADESLoss`](https://github.com/dedeswim/vits-robustness-torch/blob/master/src/attacks.py#L265). Such loss function must be a `nn.Module`, whose forward method takes as input the model, x, y, and the current epoch (used for the $\varepsilon$-warmup), and returns a tuple with, in order, the adversarial loss to differentiate, the logits given the clean data, and the logits given the adversarial data. The loss is instantiated [here](https://github.com/dedeswim/vits-robustness-torch/blob/master/src/setup_task.py#L457), so a new loss can be added after the ones that are already implemented. 

To log the results to W&B it is enough to add the flag `--log-wandb`. The W&B experiment will have the name passed to the `--experiment` flag.

### XCiT-S12, XCiT-M12, and ConvNeXt-T ImageNet training

<details>

```bash
python launch_xla.py --num-devices 8 train.py $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --output $OUTPUT --model $MODEL --config configs/xcit-adv-training.yaml
```

</details>

### XCiT-L12 ImageNet training

<details>

```bash
python launch_xla.py --num-devices 8 train.py $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --output $OUTPUT --model $MODEL --config configs/xcit-adv-training.yaml --attack-steps 2
```

</details>

### XCiT-S12, XCiT-M12 and XCiT-L12 ImageNet pre-training with epsilon 8

<details>

```bash
python launch_xla.py --num-devices 8 train.py $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --output $OUTPUT --model $MODEL --config configs/xcit-adv-training.yaml --attack-steps 2 --attack-eps 8
```

</details>

### XCiT-S12, XCiT-M12, and XCiT-L12 finetuning on Oxford Flowers or Caltech101

<details>

```bash
python launch_xla.py --num-devices 8 train.py $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --output $OUTPUT --model $MODEL --config configs/xcit-adv-finetuning-hi-res.yaml --finetune $CHECKPOINT
```

</details>


### XCiT-S12, XCiT-M12, and XCiT-L12 finetuning on Oxford Flowers or Caltech101

<details>

```bash
python launch_xla.py --num-devices 8 train.py $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --output $OUTPUT --model $MODEL --config configs/xcit-adv-finetuning-hi-res.yaml --finetune $CHECKPOINT --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --normalize-model
```

</details>

### XCiT-S12, XCiT-M12, and XCiT-L12 finetuning on CIFAR-10 and CIFAR-100

<details>

You should first download the dataset you want to finetune on with:

```bash
python3 -c "from torchvision.datasets import CIFAR10; CIFAR10('<download_dir>', download=True)"
python3 -c "from torchvision.datasets import CIFAR100; CIFAR100('<download_dir>', download=True)"
```

Then the command to start the training is:

```bash
python launch_xla.py --num-devices 8 train.py $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --output $OUTPUT --model $MODEL --config configs/xcit-adv-finetuning.yaml --finetune $CHECKPOINT --mean DATASET_MEAN --std DATASET_STD --normalize-model
```

The models are different, as we need to adapt the patch embedding layer to work on smaller resolutions. In particular, the models are:

- XCiT-S: `xcit_small_12_p4_32`
- XCiT-M: `xcit_medium_12_p4_32`
- XCiT-L: `xcit_large_12_p4_32`
- ResNet-50: `resnet_50_32`

Moreover, for CIFAR10 you should specify `--mean 0.4914 0.4822 0.4465 --std 0.2471 0.2435 0.2616`, and for CIFAR100 you should specify `--mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761`.

</details>

## Checkpoints

All the checkpoints can be found in [this](https://drive.google.com/drive/folders/1Q_E3ryzgLCkvrtmrUjal9V3Eh0X-Vpsr) Google Drive folder.

### ImageNet

#### Epsilon 4

| Model          | AutoAttack accuracy | Clean accuracy |                                         Checkpoint                                         | Model name               |
| -------------- | :-----------------: | :------------: | :----------------------------------------------------------------------------------------: | ------------------------ |
| XCiT-S12       |        41.78        |     72.34      | [link](https://drive.google.com/file/d/1wbx18o4l_ECyEYo9eyHnIYPlSNRMpO-D/view?usp=sharing) | `xcit_small_12_p16_224`  |
| XCiT-M12       |        45.24        |     74.04      | [link](https://drive.google.com/file/d/184utgGfgh6m_GDBe_mETWf0HmxksGCHH/view?usp=sharing) | `xcit_medium_12_p16_224` |
| XCiT-L12       |        47.60        |     73.76      | [link](https://drive.google.com/file/d/154l0cWMtBMK64yG7gn3fzoTQSATHR8H5/view?usp=sharing) | `xcit_large_12_p16_224`  |
| ConvNeXt-T     |        44.44        |     71.64      | [link](https://drive.google.com/file/d/1EAMVH8e67cXIFPlmK-JFFK0hzuONOWyq/view?usp=sharing) | `convnext_tiny`          |
| GELU ResNet-50 |        35.12        |     66.54      | [link](https://drive.google.com/file/d/1akb4N1B5mmesZsYGIhhDwtO-3iiKu5tb/view?usp=sharing) | `resnet_50_gelu`         |

#### Epsilon 8

| Model          | AutoAttack accuracy | Clean accuracy |                                         Checkpoint                                         | Model name               |
| -------------- | :-----------------: | :------------: | :----------------------------------------------------------------------------------------: | ------------------------ |
| XCiT-S12       |        25.00        |     63.46      | [link](https://drive.google.com/file/d/1lFcXruv6lUz9XqxvXTzyA_7YLprPZq2q/view?usp=sharing) | `xcit_small_12_p16_224`  |
| XCiT-M12       |        26.58        |     67.80      | [link](https://drive.google.com/file/d/1Nw86jzlXLEGbFQZtd39i0Mampwh2ZAFv/view?usp=sharing) | `xcit_medium_12_p16_224` |
| XCiT-L12       |        28.74        |     69.24      | [link](https://drive.google.com/file/d/1UPDBDkirsxUGBEDrznJrTZ0JR173nMlv/view?usp=sharing) | `xcit_large_12_p16_224`  |
| ConvNeXt-T     |        27.98        |     65.96      | [link](https://drive.google.com/file/d/1JyRn30kY0NPUGEigaOrGLDIkIC_V932W/view?usp=sharing) | `convnext_tiny`          |
| GELU ResNet-50 |        17.15        |     58.08      | [link](https://drive.google.com/file/d/16XlIPBlPA2xlYb5D4BOoNEgYl5UwyW7b/view?usp=sharing) | `resnet_50_gelu`         |

### CIFAR-10

| Model     | AutoAttack accuracy | Clean accuracy |                                         Checkpoint                                         | Model name             |
| --------- | :-----------------: | :------------: | :----------------------------------------------------------------------------------------: | ---------------------- |
| XCiT-S12  |        56.14        |     90.06      | [link](https://drive.google.com/file/d/1hu7Z4LhR4OOL3gmbHuT9ljRTSKIxGbPs/view?usp=sharing) | `xcit_small_12_p4_32`  |
| XCiT-M12  |        57.27        |     91.30      | [link](https://drive.google.com/file/d/1yan2zUNA6s6zDc0MOkfgCr3ctjWcQEI0/view?usp=sharing) | `xcit_medium_12_p4_32` |
| XCiT-L12  |        57.58        |     91.73      | [link](https://drive.google.com/file/d/1NWI77UiSD5nOx6dQgvIyPRYQof-8ke-y/view?usp=sharing) | `xcit_large_12_4_32`   |
| ResNet-50 |        41.56        |     84.80      | [link](https://drive.google.com/file/d/1hAHZutBd7ttO9tz30k1swc6nhyqpJqdF/view?usp=sharing) | `resnet_50_32`         |

### CIFAR-100

| Model     | AutoAttack accuracy | Clean accuracy |                                         Checkpoint                                         | Model name             |
| --------- | :-----------------: | :------------: | :----------------------------------------------------------------------------------------: | ---------------------- |
| XCiT-S12  |        32.19        |     67.34      | [link](https://drive.google.com/file/d/13vFwvVWEESfWWpoicPhD08HtzYszeyUr/view?usp=sharing) | `xcit_small_12_p4_32`  |
| XCiT-M12  |        34.21        |     69.21      | [link](https://drive.google.com/file/d/1YXnGsY3yvvaMucMwyfhDHIUchOSungj2/view?usp=sharing) | `xcit_medium_12_p4_32` |
| XCiT-L12  |        35.08        |     70.76      | [link](https://drive.google.com/file/d/1Sg4BQ5pBdlR_M41VDxOlYxCl1_It8elL/view?usp=sharing) | `xcit_large_12_4_32`   |
| ResNet-50 |        22.01        |     61.28      | [link](https://drive.google.com/file/d/1RweWjpjMyyhF8P1JdjuDc09F6wp4TsF9/view?usp=sharing) | `resnet_50_32`         |

### Oxford Flowers

| Model     | AutoAttack accuracy | Clean accuracy |                                         Checkpoint                                         | Model name              |
| --------- | :-----------------: | :------------: | :----------------------------------------------------------------------------------------: | ----------------------- |
| XCiT-S12  |        47.91        |     82.86      | [link](https://drive.google.com/file/d/1XHCbdj5vXUybofNmNd5a2LGQeVxITCpE/view?usp=sharing) | `xcit_small_12_p12_224` |
| ResNet-50 |        32.75        |     74.51      | [link](https://drive.google.com/file/d/1y41kNkiSwNPNs61QZhm9Cy0arBkeKjA0/view?usp=sharing) | `resnet_50`             |

### Caltech101

| Model     | AutoAttack accuracy | Clean accuracy |                                         Checkpoint                                         | Model name              |
| --------- | :-----------------: | :------------: | :----------------------------------------------------------------------------------------: | ----------------------- |
| XCiT-S12  |        61.74        |     87.59      | [link](https://drive.google.com/file/d/1l_wdzgXI5lE6RVOtZvIzaPOMfoL6a1GJ/view?usp=sharing) | `xcit_small_12_p12_224` |
| ResNet-50 |        34.49        |     81.38      | [link](https://drive.google.com/file/d/1sNwwZ4gIP7yIwv87RSaWu70duAUotruW/view?usp=sharing) | `resnet_50`             |

## Validation

For validating using full AA models trained on ImageNet, CIFAR-10 and CIFAR-100 it is recommended to use [this](#validating-using-robustbench) command. To evaluate using APGD-CE only, or to evaluate other datasets than those above (e.g., Caltech101 and Oxford Flowers), then use [this](#validating-using-the-validation-script) script instead.

### Validating using RobustBench

<details>

This script will run the full AutoAttack using RobustBench's interface.

```bash
python3 validate_robustbench.py --data-dir $DATA_DIR --dataset $DATASET --model $MODEL --batch-size 1024 --checkpoint $CHECKPOINT --eps $EPS
```

If the model has been trained using a specific mean and std, then they should be specified with the `--mean` and `--std` flags, similarly to training.

</details>

### Validating using the validation script

Do not use this script to run APGD-CE or AutoAttack on TPU (and XLA in general), as the compilation will take an unreasonable amount of time.

<details>

```bash
python3 validate.py $DATA_DIR --dataset $DATASET --log-freq 1 --model $MODEL --checkpoint $CHECKPOINT --mean <mean> --std <std> --attack $ATTACK --attack-eps $EPS
```

If the model has been trained using a specific mean and std, then they should be specified with the `--mean` and `--std` flags, and the `--normalize-model` flag should be specified, similarly to training. Otherwise the `--no-normalize` flag sould be specified. For both Caltech101 and Oxford Flowers, you should specify `--num-classes 102`, and for Caltech101 only `--split test`. If you just want to run PGD, then you can specify the number of steps with `--attack-steps 200`.

</details>

## Attack effectiveness experiment

To reproduce the attack effectiveness experiment, you can run the [`attack_effectiveness.py`](attack_effectiveness.py) script. The results are written to a CSV file created in the same folder as that of the checkpoints that are tested. We process the CSV files generated with the [attack_effectiveness.ipynb](notebooks/attack_effectiveness.ipynb) notebook. For instance, the validation of XCiT-S12 checkpoints, assuming that in the folder `checkpoints` the checkpoints are in the form `checkpoint-{epoch}-pth.tar`, the script can be launched as

```bash
DATA_DIR=... \ # Location of the TFDS data or the torch data
DATASET=tfds/robustbench_image_net \ # or any other dataset, both torch and tfds
MODEL=xcit_small_12_p16_224 \ # Or any other model
CHECKPOINTS_DIR=checkpoints \ # The checkpoint to validate
EPS=8 \ # The epsilon to use to evaluate
python3 attack_effectiveness.py $DATA_DIR --dataset $DATASET --model $MODEL --checkpoints-dir $CHECKPOINTS_DIR --no-normalize
```

This will run PGD with 1, 2, 5, 10, and 200 steps, for the checkpoints at the epochs 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, with three different seeds. The results will be written in `checkpoints/summary.csv`.

The checkpoints for the models we test with this experiments (XCiT-S12, ResNet-50 GELU, and ConvNeXt-T) can be found at [this](https://drive.google.com/drive/u/0/folders/1PhGnEhM5nVG_L23R6VZqiLM-iwKHbWA8) Google Drive link.

## Code

A large amount of the code is adapted from [`timm`](https://github.com/rwightman/pytorch-image-models), in particular from the `bits_and_tpu` branch. The code by Ross Wightman is originally released under Apache-2.0 License, which can be found [here](https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE).

The entry point for training is [train.py](train.py). While in [src](src/) there is a bunch of utility modules, as well as model definitions (which are found in [src/models](src/models/)).

### Datasets directory

The [datasets](datasets/) directory contains the code to generate the TFDS datasets for:
- [CIFAR 10 synthetic data](datasets/cifar_ddpm/) (from https://arxiv.org/abs/2104.09425): to mix with CIFAR-10
- [ImageNet Subset](datasets/image_net_subset/): to generate the ImageNet subset of 100 classes used for the ablation.
- [RobustBench Imagenet](datasets/robustbench_image_net/): to generate the subset of 5000 images used in RobustBench as a TFDS dataset.
- [ImageNet Perturbations](datasets/imagenet_perturbations/): to generate a dataset of adversarial perturbations targeting several models for the RobustBench subset. We used these perturbations to classify them with SOTA ImageNet models to quantify the perceptual nature of adversarial perturbations.
- [ImageNet AdvEx](datasets/imagenet_advex/): to generate a dataset of adversarial examples targeting several models for the RobustBench subset.

Additional information about how TFDS datasets are generated can be found on TFDS' [documentation](https://www.tensorflow.org/datasets/add_dataset).

### Tests

In order to run the unit tests in the repo, install pytest via `pip install pytest`, and run

```bash
python -m pytest .
```

## Acknowledgments

As mentioned above, this codebase is based on `timm`, we thank @rwightman for sharing it. Moreover, we thank Google’s [TPU Research Cloud (TRC) program](https://sites.research.google/trc/about/), which provided us with extremely generous computing resources which enabled us to train the models we are sharing.

## Citation

If you find our work useful, please consider citing it using the following BibTeX entry:

```bibtex
@misc{debenedetti2022light,
  doi = {10.48550/ARXIV.2209.07399},
  url = {https://arxiv.org/abs/2209.07399},
  author = {Debenedetti, Edoardo and Sehwag, Vikash and Mittal, Prateek},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Light Recipe to Train Robust Vision Transformers},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

import argparse
import math

import torch
from robustbench import benchmark
from timm.models import xcit
from torchvision import transforms
from torch import nn

import src.models as models
import src.utils as utils

from validate import log_results_to_wandb

parser = argparse.ArgumentParser(description='Validation using RobustBench')
parser.add_argument('--patch-size', default=16, type=int, metavar='N', help='The patch size to use')
parser.add_argument('--dataset', default="cifar10", type=str, metavar='DATASET')
parser.add_argument('--data-dir', default="~/torch_data/", type=str, metavar='DATASET')
parser.add_argument('--crop-pct', default=None, type=float)
parser.add_argument('--normalize', action='store_true', default=False, help='Normalizes inputs')
parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH')
parser.add_argument('--model', default=None, type=str, metavar='NAME')
parser.add_argument('--batch-size', default=512, type=int, metavar='N')
parser.add_argument('--num-classes', default=10, type=int, metavar='N')
parser.add_argument('--eps', default=8, type=int)
parser.add_argument('--threat-model', default="Linf", type=str, metavar='MODEL')
parser.add_argument('--log-wandb',
                    action='store_true',
                    default=False,
                    help='Log results to wandb using the run stored in the bucket')
parser.add_argument('--upsample-in-model', action='store_true', default=False, help='')
parser.add_argument('--mean',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='STD',
                    help='Override std deviation of of dataset')


def main(args):
    model = utils.load_model_from_gcs(
        args.checkpoint,
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
    )
    if isinstance(model, xcit.XCiT) and model.patch_embed.patch_size != args.patch_size:
        assert args.patch_size in {2, 4, 8}, "Finetuning patch size can be only 4, 8 or `None`"
        assert isinstance(model, xcit.XCiT), "Finetuning patch size is only supported for XCiT"
        print(f"Adapting patch embedding for finetuning patch size {args.patch_size}")
        model.patch_embed.patch_size = args.patch_size
        model.patch_embed.proj[0][0].stride = (1, 1)
        if args.patch_size == 4:
            model.patch_embed.proj[2][0].stride = (1, 1)
        if args.patch_size == 2:
            model.patch_embed.proj[4][0].stride = (1, 1)

    device = torch.device("cuda:0")

    # Get default pre-processing settings from the model
    interpolation = model.default_cfg['interpolation']
    crop_pct = args.crop_pct or model.default_cfg['crop_pct']
    img_size = model.default_cfg['input_size'][1]
    scale_size = int(math.floor(img_size / crop_pct))
    if args.dataset == "imagenet":
        preprocessing = transforms.Compose([
            transforms.Resize(scale_size, interpolation=transforms.InterpolationMode(interpolation)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
        if args.normalize:
            preprocessing.transforms.append(
                transforms.Normalize(model.default_cfg['mean'], model.default_cfg['std']))
    else:
        preprocessing = transforms.ToTensor()

    if args.mean is not None or args.std is not None:
        mean = args.mean or (0, 0, 0)
        std = args.std or (1, 1, 1)
        model = utils.normalize_model(model, mean=mean, std=std)

    if args.upsample_in_model:
        model = nn.Sequential(
            transforms.Resize(scale_size, interpolation=transforms.InterpolationMode(interpolation)),
            transforms.CenterCrop(img_size), model)

    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    clean_acc, robust_acc = benchmark(model,
                                      dataset=args.dataset,
                                      data_dir=args.data_dir,
                                      device=device,
                                      batch_size=args.batch_size,
                                      eps=args.eps / 255,
                                      preprocessing=preprocessing,
                                      #n_examples=256,
                                      threat_model=args.threat_model)

    if args.log_wandb:
        args.attack_eps = args.eps / 255
        args.attack_steps = None
        args.attack = "autoattack"
        args.attack_norm = args.threat_model.lower()  # .lower() to comply with training convention
        results_dict = {'top1': clean_acc * 100, 'robust_top1': robust_acc * 100}
        log_results_to_wandb(args, results_dict)


if __name__ == "__main__":
    _args = parser.parse_args()
    main(_args)

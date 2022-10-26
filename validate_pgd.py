import argparse
import csv

from timm.utils import setup_default_logging

import src.models as models  # Import needed to register the extra models that are not in timm
from validate import validate, write_results

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset',
                    '-d',
                    metavar='NAME',
                    default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split',
                    metavar='NAME',
                    default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download',
                    action='store_true',
                    default=False,
                    help='Allow download of dataset for torch/ '
                    'and tfds/ datasets that support it.')
parser.add_argument('--model',
                    '-m',
                    metavar='NAME',
                    default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--img-size',
                    default=None,
                    type=int,
                    metavar='N',
                    help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size',
                    default=None,
                    nargs=3,
                    type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), '
                    'uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float, metavar='N', help='Input image center crop pct')
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
parser.add_argument('--interpolation',
                    default='',
                    type=str,
                    metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-nn',
                    '--no-normalize',
                    action='store_true',
                    default=False,
                    help='Avoids normalizing inputs (but it scales them in [0, 1]')
parser.add_argument('--normalize-model',
                    action='store_true',
                    default=False,
                    help='Performs normalization as part of the model')
parser.add_argument('--num-classes', type=int, default=None, help='Number classes in dataset')
parser.add_argument('--class-map',
                    default='',
                    type=str,
                    metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp',
                    default=None,
                    type=str,
                    metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq',
                    default=1,
                    type=int,
                    metavar='N',
                    help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
# parser.add_argument('--num-gpu', type=int, default=1,
#                     help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true', help='enable test time pool')
parser.add_argument('--pin-mem',
                    action='store_true',
                    default=False,
                    help='Pin CPU memory in DataLoader for more'
                    'efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last',
                    action='store_true',
                    default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp',
                    action='store_true',
                    default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--tf-preprocessing',
                    action='store_true',
                    default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema',
                    dest='use_ema',
                    action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript',
                    dest='torchscript',
                    action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results-file',
                    default='',
                    type=str,
                    metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels',
                    default='',
                    type=str,
                    metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels',
                    default='',
                    type=str,
                    metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--force-cpu',
                    action='store_true',
                    default=False,
                    help='Force CPU to be used even if HW accelerator exists.')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')

parser.add_argument('--attack',
                    default='',
                    type=str,
                    metavar='ATTACK',
                    help='What attack to use (default: "pgd")')
parser.add_argument('--attack-eps',
                    default=4,
                    type=float,
                    metavar='EPS',
                    help='The epsilon to use for the attack (default 4/255)')
parser.add_argument('--attack-lr',
                    default=None,
                    type=float,
                    metavar='ATTACK_LR',
                    help='Learning rate for the attack (default 1e-4)')
parser.add_argument('--attack-steps',
                    default=10,
                    type=int,
                    nargs='+',
                    metavar='ATTACK_STEPS',
                    help='Number of steps to run attack for (default 10)')
parser.add_argument('--attack-norm',
                    default='linf',
                    type=str,
                    metavar='NORM',
                    help='The norm to use for the attack (default linf)')
parser.add_argument('--attack-boundaries',
                    default=(0, 1),
                    nargs=2,
                    type=int,
                    metavar='L H',
                    help='Boundaries of projection')
parser.add_argument('--log-wandb',
                    action='store_true',
                    default=False,
                    help='Log results to wandb using the run stored in the bucket')
parser.add_argument('--use-mp-loader', action='store_true', default=False, help='Use Torch XLA\'s  MP Loader')
parser.add_argument('--num-examples',
                    default=None,
                    type=int,
                    metavar='EXAMPLES',
                    help='Number of examples to use for the evaluation (default the entire dataset)')
parser.add_argument('--patch-size', default=None, type=int, metavar='N', help='The patch size to use')
parser.add_argument('--verbose', action='store_true', default=False, help='Runs autoattack in verbose mode')


def main():
    setup_default_logging()
    args = parser.parse_args()
    steps_to_try = args.attack_steps

    results_file = args.results_file or './results-all.csv'
    all_results = []

    for steps in steps_to_try:
        args.attack_steps = steps
        steps_results = validate(args)
        steps_results["attack_steps"] = steps
        all_results.append(steps_results)

    write_results(results_file, all_results)

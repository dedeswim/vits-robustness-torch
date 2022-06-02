"""Argumet parser for training.

Parts of this file are adapted from PyTorch Image Models by Ross Wightman

The original ones can be found at https://github.com/rwightman/pytorch-image-models/

The original license can be found at this link: https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE
"""

import argparse
import yaml

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='',
                    type=str,
                    metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset',
                    '-d',
                    metavar='NAME',
                    default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--combine-dataset',
                    metavar='NAME',
                    default=None,
                    help='Combine a dataset to the original one')
parser.add_argument('--combine-data-dir',
                    metavar='NAME',
                    default=None,
                    help='Directory of the dataset to combine')
parser.add_argument('--combined-dataset-ratio',
                    metavar='F',
                    type=float,
                    default=0.5,
                    help='Ratio of the combined dataset, default is 0.5')
parser.add_argument('--train-split',
                    metavar='NAME',
                    default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split',
                    metavar='NAME',
                    default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model',
                    default='resnet50',
                    type=str,
                    metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained',
                    action='store_true',
                    default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt',
                    action='store_true',
                    default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes',
                    type=int,
                    default=None,
                    metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp',
                    default=None,
                    type=str,
                    metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size',
                    type=int,
                    default=None,
                    metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument(
    '--input-size',
    default=None,
    nargs=3,
    type=int,
    metavar='N N N',
    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct',
                    default=None,
                    type=float,
                    metavar='N',
                    help='Input image center crop percent (for validation only)')
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
parser.add_argument('-b',
                    '--batch-size',
                    type=int,
                    default=256,
                    metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb',
                    '--validation-batch-size',
                    type=int,
                    default=None,
                    metavar='N',
                    help='validation batch size override (default: None)')
parser.add_argument('-nn',
                    '--no-normalize',
                    action='store_true',
                    default=True,
                    help='Avoids normalizing inputs (but it scales them in [0, 1]')
parser.add_argument('--normalize-model',
                    action='store_true',
                    default=False,
                    help='Applies normalization as part of the model')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER', help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps',
                    default=None,
                    type=float,
                    metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas',
                    default=None,
                    type=float,
                    nargs='+',
                    metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad',
                    type=float,
                    default=None,
                    metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode',
                    type=str,
                    default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched',
                    default='cosine',
                    type=str,
                    metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.05)')
parser.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                    help='base learning rate: lr = lr_base * global_batch_size / base_size')
parser.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                    help='base learning rate batch size (divisor, default: 256).')
parser.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
parser.add_argument('--lr-noise',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct',
                    type=float,
                    default=0.67,
                    metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std',
                    type=float,
                    default=1.0,
                    metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul',
                    type=float,
                    default=1.0,
                    metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay',
                    type=float,
                    default=0.5,
                    metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit',
                    type=int,
                    default=1,
                    metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay',
                    type=float,
                    default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr',
                    type=float,
                    default=0.0001,
                    metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr',
                    type=float,
                    default=1e-5,
                    metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs',
                    type=int,
                    default=300,
                    metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats',
                    type=float,
                    default=0.,
                    metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch',
                    default=None,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs',
                    type=int,
                    default=5,
                    metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs',
                    type=int,
                    default=10,
                    metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs',
                    type=int,
                    default=10,
                    metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate',
                    '--dr',
                    type=float,
                    default=0.1,
                    metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug',
                    action='store_true',
                    default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale',
                    type=float,
                    nargs='+',
                    default=[0.08, 1.0],
                    metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio',
                    type=float,
                    nargs='+',
                    default=[3. / 4., 4. / 3.],
                    metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0., help='Vertical flip training aug probability')
parser.add_argument('--color-jitter',
                    type=float,
                    default=0.4,
                    metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa',
                    type=str,
                    default=None,
                    metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits',
                    type=int,
                    default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss',
                    action='store_true',
                    default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss',
                    action='store_true',
                    default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh',
                    type=float,
                    default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT', help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
parser.add_argument('--resplit',
                    action='store_true',
                    default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup',
                    type=float,
                    default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix',
                    type=float,
                    default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax',
                    type=float,
                    nargs='+',
                    default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob',
                    type=float,
                    default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob',
                    type=float,
                    default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode',
                    type=str,
                    default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation',
                    type=str,
                    default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect',
                    type=float,
                    default=None,
                    metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path',
                    type=float,
                    default=None,
                    metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block',
                    type=float,
                    default=None,
                    metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-momentum',
                    type=float,
                    default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None, help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn',
                    action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument(
    '--dist-bn',
    type=str,
    default='reduce',
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn',
                    action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema',
                    action='store_true',
                    default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-decay',
                    type=float,
                    default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--log-interval',
                    type=int,
                    default=50,
                    metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval',
                    type=int,
                    default=0,
                    metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist',
                    type=int,
                    default=10,
                    metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j',
                    '--workers',
                    type=int,
                    default=4,
                    metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images',
                    action='store_true',
                    default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp',
                    action='store_true',
                    default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--channels-last',
                    action='store_true',
                    default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem',
                    action='store_true',
                    default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--output',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment',
                    default='',
                    type=str,
                    metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric',
                    default='top1',
                    type=str,
                    metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta',
                    type=int,
                    default=0,
                    metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader',
                    action='store_true',
                    default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript',
                    dest='torchscript',
                    action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--force-cpu',
                    action='store_true',
                    default=False,
                    help='Force CPU to be used even if HW accelerator exists.')
parser.add_argument('--log-wandb',
                    action='store_true',
                    default=False,
                    help='log training and validation metrics to wandb')
parser.add_argument('--run-notes', default='', type=str, help='Description about this run')

# Adversarial training arguments
# Args for adversarial training:
parser.add_argument('--adv-training',
                    default=None,
                    type=str,
                    help='Enables adversarial training with the specified '
                    'technique (`trades` or `pgd`)')
parser.add_argument('--attack',
                    default='pgd',
                    type=str,
                    metavar='ATTACK',
                    help='What attack to use (default: "pgd")')
parser.add_argument('--attack-eps',
                    default=4,
                    type=float,
                    metavar='EPS',
                    help='The epsilon to use for the attack (default 4/255)')
parser.add_argument('--eps-schedule',
                    default='constant',
                    type=str,
                    metavar='SCHEDULE',
                    help='What schedule to use for eps (default: "constant")')
parser.add_argument('--eps-schedule-period',
                    default=10,
                    type=int,
                    metavar='EPOCHS',
                    help='How many epochs before reaching full eps')
parser.add_argument('--zero-eps-epochs',
                    default=0,
                    type=int,
                    metavar='EPOCHS',
                    help='How many epochs eps should be 0')
parser.add_argument('--attack-lr',
                    default=None,
                    type=float,
                    metavar='ATTACK_LR',
                    help='Learning rate for the attack (default 1e-4)')
parser.add_argument('--attack-steps',
                    default=10,
                    type=int,
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
parser.add_argument('--eval-attack-eps',
                    default=None,
                    type=float,
                    metavar='EPS',
                    help='The epsilon to use for the attack (default the same as `--attack-eps`)')
parser.add_argument('--trades-beta', default=6.0, type=float)

parser.add_argument('--finetune', default=None, type=str, help='Finetune from checkpoint')
parser.add_argument(
    '--finetuning-patch-size',
    default=None,
    type=int,
    metavar='X',
    help='Patch size to use for fine-tuning (can be only 4 or 8). If None, the original patch size is used.')
parser.add_argument('--reinit-patch-embedding',
                    action='store_true',
                    default=False,
                    help='Re-initializes the whole patch embedder')
parser.add_argument('--keep-patch-embedding',
                    action='store_true',
                    default=False,
                    help='Re-initializes the whole patch embedder')
parser.add_argument('--use-mp-loader',
                    action='store_true',
                    default=False,
                    help='Uses torch_xla\'s MP loader')


def parse_args(additional_args=None):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.    
    if additional_args is not None:
        remaining += additional_args
    args = parser.parse_args(remaining)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

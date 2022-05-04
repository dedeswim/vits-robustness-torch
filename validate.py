#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script
for evaluating pretrained models or training checkpoints against ImageNet or
similarly organized image datasets. It prioritizes canonical PyTorch, standard
Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import csv
import glob
import logging
import math
import os
from collections import OrderedDict
from typing import Dict

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.parallel
import yaml
from timm.bits import (AccuracyTopK, AvgTensor, Monitor, Tracker, initialize_device)
from timm.data import (RealLabelsImagenet, create_dataset, create_loader_v2, resolve_data_config)
from timm.models import (apply_test_time_pool, create_model, is_model, list_models, load_checkpoint, xcit)
from timm.utils import natural_key, setup_default_logging
from torchvision import transforms

import src.attacks as attacks
import src.models as models  # Import needed to register the extra models that are not in timm
import src.utils as utils

_logger = logging.getLogger('validate')

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
parser.add_argument('--num-examples',
                    default=None,
                    type=int,
                    metavar='EXAMPLES',
                    help='Number of examples to use for the evaluation (default the entire dataset)')
parser.add_argument('--patch-size', default=None, type=int, metavar='N', help='The patch size to use')
parser.add_argument('--verbose', action='store_true', default=False, help='Runs autoattack in verbose mode')


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint

    dev_env = initialize_device(force_cpu=args.force_cpu, amp=args.amp)

    model = create_model(args.model,
                         pretrained=args.pretrained,
                         num_classes=args.num_classes,
                         in_chans=3,
                         global_pool=args.gp,
                         scriptable=args.torchscript)

    if args.num_classes is None:
        assert hasattr(model,
                       'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint.startswith('gs://'):
        model = utils.load_model_from_gcs(args.checkpoint,
                                          args.model,
                                          pretrained=args.pretrained,
                                          num_classes=args.num_classes,
                                          in_chans=3,
                                          global_pool=args.gp,
                                          scriptable=args.torchscript)
    elif args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    if args.patch_size is not None and isinstance(
            model, xcit.XCiT) and model.patch_embed.patch_size != args.patch_size:
        assert args.patch_size in {2, 4, 8}, "Finetuning patch size can be only 4, 8 or `None`"
        assert isinstance(model, models.xcit.XCiT), "Finetuning patch size is only supported for XCiT"
        _logger.info(f"Adapting patch embedding for finetuning patch size {args.patch_size}")
        model.patch_embed.patch_size = args.patch_size
        model.patch_embed.proj[0][0].stride = (1, 1)
        if args.patch_size == 4:
            model.patch_embed.proj[2][0].stride = (1, 1)
        if args.patch_size == 2:
            model.patch_embed.proj[4][0].stride = (1, 1)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)
    data_config['normalize'] = not (args.no_normalize or args.normalize_model)

    if args.normalize_model:
        mean = args.mean or data_config["mean"]
        std = args.std or data_config["std"]
        model = utils.normalize_model(model, mean=mean, std=std)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    # FIXME device
    model, criterion = dev_env.to_device(model, nn.CrossEntropyLoss())
    model.to(dev_env.device)

    dataset = create_dataset(root=args.data,
                             name=args.dataset,
                             split=args.split,
                             download=args.dataset_download,
                             load_bytes=args.tf_preprocessing,
                             class_map=args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    eval_pp_cfg = utils.MyPreprocessCfg(  # type: ignore
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        crop_pct=1.0 if test_time_pool else data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        normalize=data_config['normalize'],
    )

    loader = create_loader_v2(dataset,
                              batch_size=args.batch_size,
                              is_training=False,
                              pp_cfg=eval_pp_cfg,
                              num_workers=args.workers,
                              pin_memory=args.pin_mem)

    if not eval_pp_cfg.normalize:
        loader.dataset.transform.transforms[-1] = transforms.ToTensor()

    logger = Monitor(logger=_logger)
    tracker = Tracker()
    losses = AvgTensor()
    adv_losses = AvgTensor()
    accuracy = AccuracyTopK(dev_env=dev_env)
    adv_accuracy = AccuracyTopK(dev_env=dev_env)

    if args.attack:
        eps = args.attack_eps / 255
        lr = args.attack_lr or (1.5 * eps / args.attack_steps)
        attack_criterion = nn.NLLLoss(reduction="sum")
        attack_kwargs = {}
        if args.attack in {"autoattack", "apgd-ce"}:
            attack_kwargs["verbose"] = args.verbose
        else:
            attack_kwargs["dev_env"] = dev_env
        attack = attacks.make_attack(args.attack, eps, lr, args.attack_steps, args.attack_norm,
                                     args.attack_boundaries, attack_criterion, **attack_kwargs)
    else:
        attack = None

    model.eval()
    num_steps = len(loader)
    if args.num_examples is not None:
        num_steps = min(num_steps, args.num_examples // args.batch_size)
        print(f"Total steps: {num_steps}")

    with torch.no_grad():
        tracker.mark_iter()
        for step_idx, (sample, target) in enumerate(loader):
            last_step = step_idx == num_steps - 1
            tracker.mark_iter_data_end()

            with dev_env.autocast():
                if attack is not None:
                    if dev_env.type_xla:
                        model.train()
                    with torch.enable_grad():
                        adv_sample = attack(model, sample, target)
                    model.eval()
                    adv_output = model(adv_sample)
                else:
                    adv_output = None
                output = model(sample)
            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)
            if adv_output is not None:
                adv_loss = criterion(adv_output, target)
            else:
                adv_loss = None

            if dev_env.type_xla:
                dev_env.mark_step()
            elif dev_env.type_cuda:
                dev_env.synchronize()
            tracker.mark_iter_step_end()

            if real_labels is not None:
                real_labels.add_result(output)
            losses.update(loss.detach(), sample.size(0))
            accuracy.update(output.detach(), target)
            if adv_output is not None:
                adv_accuracy.update(adv_output.detach(), target)
            if adv_losses is not None:
                adv_losses.update(adv_loss.detach(), sample.size(0))

            tracker.mark_iter()
            if last_step or step_idx % args.log_freq == 0:
                top1, top5 = accuracy.compute().values()
                robust_top1, robust_top5 = adv_accuracy.compute().values()
                loss_avg = losses.compute()
                adv_loss_avg = adv_losses.compute()

                logger.log_step(
                    phase='eval',
                    step_idx=step_idx,
                    num_steps=num_steps,
                    rate=(tracker.get_last_iter_rate(output.shape[0]),
                          tracker.get_avg_iter_rate(args.batch_size)),
                    loss=loss_avg.item(),
                    top1=top1.item(),
                    top5=top5.item(),
                    adv_loss=adv_loss_avg.item(),
                    robust_top1=robust_top1.item(),
                    robust_top5=robust_top5.item(),
                )

            if last_step:
                break

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = accuracy.compute().values()
        top1a, top5a = top1a.item(), top5a.item()
        robust_top1a, robust_top5a = adv_accuracy.compute().values()
        robust_top1a, robust_top5a = robust_top1a.item(), robust_top5a.item()

    results = OrderedDict(top1=round(top1a, 4),
                          top1_err=round(100 - top1a, 4),
                          top5=round(top5a, 4),
                          top5_err=round(100 - top5a, 4),
                          robust_top1=round(robust_top1a, 4),
                          robust_top1_err=round(100 - robust_top1a, 4),
                          robust_top5=round(robust_top5a, 4),
                          robust_top5_err=round(100 - robust_top5a, 4),
                          param_count=round(param_count / 1e6, 2),
                          img_size=data_config['input_size'][-1],
                          cropt_pct=eval_pp_cfg.crop_pct,
                          interpolation=data_config['interpolation'])
    logger.log_phase(phase='eval',
                     name_map={
                         'top1': 'Acc@1',
                         'top5': 'Acc@5',
                         'robust_top1': 'RobustAcc@1',
                         'robust_top5': 'RobustAcc@5',
                     },
                     **results)

    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= 1:
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        results = validate(args)
        if args.log_wandb:
            log_results_to_wandb(args, results)


def log_results_to_wandb(args: argparse.Namespace, results: Dict):
    import wandb

    # Get args file from bucket
    assert args.checkpoint.startswith('gs://')
    experiment_dir = os.path.dirname(args.checkpoint)
    args_path = os.path.join(experiment_dir, 'args.yaml')
    with tf.io.gfile.GFile(args_path, mode='r') as f:
        config = yaml.safe_load(f)
    wandb_run_url = config["wandb_run"]
    # Get run identifying info
    if wandb_run_url.endswith('/'):
        wandb_run_url = wandb_run_url[:-1]
    wandb_run_project = wandb_run_url.split("/")[4]
    wandb_run_entity = wandb_run_url.split("/")[3]
    wandb_run_id = wandb_run_url.split("/")[6]
    run = wandb.init(project=wandb_run_project, id=wandb_run_id, entity=wandb_run_entity, resume=True)
    # Log data
    attack = args.attack
    eps = args.attack_eps
    steps = args.attack_steps
    norm = args.attack_norm
    prefix = f"{attack}-{steps}-{eps}"
    if norm != "linf":
        prefix += f"-{norm}"
    dict_to_log = {
        "eval_top1-final": results['top1'],
        f"{prefix}-eval_robust_top1_final": results['robust_top1'],
    }
    run.log(dict_to_log)
    run.finish()


def write_results(results_file, results):
    if results_file.startswith("gs://"):
        open_f = tf.io.gfile.GFile
    else:
        open_f = open
    with open_f(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=list(results[0].keys()))
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


def _mp_entry(*args):
    main()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import logging
import os
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime
from typing import Optional, Tuple

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from timm.bits import (AccuracyTopK, AvgTensor, CheckpointManager, DeviceEnv, Monitor, Tracker, TrainCfg,
                       TrainServices, TrainState, distribute_bn, initialize_device, setup_model_and_optimizer)
from timm.data import (AugCfg, AugMixDataset, fetcher, MixupCfg, create_loader_v2, resolve_data_config)
from timm.data.dataset_factory import create_dataset
from timm.loss import (BinaryCrossEntropy, JsdCrossEntropy, LabelSmoothingCrossEntropy,
                       SoftTargetCrossEntropy)
from timm.models import convert_splitbn_model, create_model, safe_model_name
from timm.optim import optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import random_seed, setup_default_logging, unwrap_model
from torchvision import transforms

from src.iterable_augmix_dataset import IterableAugMixDataset
from src import attacks, models, utils  # Models import needed to register the extra models that are not in timm
from src.arg_parser import parse_args
from src.attacks import _SCHEDULES, AttackFn
from src.random_erasing import NotNormalizedRandomErasing

_logger = logging.getLogger('train')


def main():
    setup_default_logging()
    args, args_text = parse_args()

    dev_env = initialize_device(force_cpu=args.force_cpu, amp=args.amp, channels_last=args.channels_last)
    if dev_env.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 device per process. '
                     'Process %d, total %d.' % (dev_env.global_rank, dev_env.world_size))
    else:
        _logger.info('Training with a single process on 1 device.')

    random_seed(args.seed, 0)  # Set all random seeds the same for model/state init (mandatory for XLA)

    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    assert args.aug_splits == 0 or args.aug_splits > 1, 'A split of 1 makes no sense'

    train_state = setup_train_task(args, dev_env, mixup_active)
    train_cfg = train_state.train_cfg

    # Set random seeds across ranks differently for train
    # FIXME perhaps keep the same and just set diff seeds for dataloader worker process? what about TFDS?
    random_seed(args.seed, dev_env.global_rank)

    data_config, loader_eval, loader_train = setup_data(args,
                                                        unwrap_model(train_state.model).default_cfg, dev_env,
                                                        mixup_active)

    if args.normalize_model:
        train_state = replace(train_state,
                              model=utils.normalize_model(train_state.model,
                                                          mean=data_config["mean"],
                                                          std=data_config["std"]))
        train_state = replace(train_state, model=dev_env.to_device(train_state.model))
        if train_state.model_ema is not None:
            train_state = replace(train_state,
                                  model_ema=utils.normalize_model(train_state.model_ema,
                                                                  mean=data_config["mean"],
                                                                  std=data_config["std"]))
            train_state = replace(train_state, model_ema=dev_env.to_device(train_state.model_ema))

    # setup checkpoint manager
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    checkpoint_manager = None
    output_dir = None
    checkpoints_dir = None
    if dev_env.primary:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name, inc=True)
        if output_dir.startswith("gs://"):
            checkpoints_dir = utils.get_outdir('./output/tmp/', exp_name, inc=True)
            _logger.info(f"Temporarily saving checkpoints in {checkpoints_dir}")
        else:
            checkpoints_dir = output_dir
        checkpoint_manager = CheckpointManager(hparams=vars(args),
                                               checkpoint_dir=checkpoints_dir,
                                               recovery_dir=output_dir,
                                               metric_name=eval_metric,
                                               metric_decreasing=True if eval_metric == 'loss' else False,
                                               max_history=args.checkpoint_hist)

        if output_dir.startswith("gs://"):
            with tf.io.gfile.GFile(os.path.join(output_dir, 'args.yaml'), 'w') as f:
                f.write(args_text)
        else:
            with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
                f.write(args_text)

    services = TrainServices(
        monitor=Monitor(output_dir=output_dir,
                        logger=_logger,
                        hparams=vars(args),
                        output_enabled=dev_env.primary,
                        experiment_name=args.experiment,
                        log_wandb=args.log_wandb and dev_env.primary),
        checkpoint=checkpoint_manager,  # type: ignore
    )

    if (wandb_run := services.monitor.wandb_run) is not None:
        assert output_dir is not None
        # Log run notes and *true* output dir to wandb
        notes = args.run_notes
        if output_dir.startswith("gs://"):
            exp_dir = output_dir.split("gs://")[-1]
            bucket_url = f"https://console.cloud.google.com/storage/{exp_dir}"
            notes += f"Bucket: {exp_dir}\n"
            wandb_run.config.update({"output": bucket_url}, allow_val_change=True)
        else:
            wandb_run.config.update({"output": output_dir}, allow_val_change=True)
        wandb_run.notes = notes
        wandb_run_field = f"wandb_run: {wandb_run.url}\n"  # type: ignore
        # Log wandb run url to args file
        if output_dir.startswith("gs://"):
            with tf.io.gfile.GFile(os.path.join(output_dir, 'args.yaml'), 'a') as f:
                f.write(wandb_run_field)
        else:
            with open(os.path.join(output_dir, 'args.yaml'), 'a') as f:
                f.write(wandb_run_field)

    if output_dir is not None and output_dir.startswith('gs://'):
        services.monitor.csv_writer = utils.GCSSummaryCsv(output_dir=output_dir)

    if args.adv_training is not None:
        attack_criterion = nn.NLLLoss(reduction="sum")
        dev_env.to_device(attack_criterion)
        eval_attack_name = args.attack.split("targeted_")[-1]
        eps = (args.eval_attack_eps or args.attack_eps) / 255
        attack_step_size = args.attack_lr or (1.5 * eps / args.attack_steps)
        eval_attack = attacks.make_attack(eval_attack_name,
                                          eps,
                                          attack_step_size,
                                          args.attack_steps,
                                          args.attack_norm,
                                          args.attack_boundaries,
                                          criterion=attack_criterion)
    else:
        eval_attack = None

    _logger.info('Starting training, the first steps may take a long time')

    try:
        for epoch in range(train_state.epoch, train_cfg.num_epochs):
            if dev_env.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
                if loader_train.mixup_enabled:
                    loader_train.mixup_enabled = False

            train_metrics = train_one_epoch(
                state=train_state,
                services=services,
                loader=loader_train,
                dev_env=dev_env,
            )

            if dev_env.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if dev_env.primary:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(train_state.model, args.dist_bn == 'reduce', dev_env)

            eval_metrics = evaluate(train_state.model,
                                    train_state.eval_loss,
                                    loader_eval,
                                    services.monitor,
                                    dev_env,
                                    attack=eval_attack)

            if train_state.model_ema is not None:
                if dev_env.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(train_state.model_ema, args.dist_bn == 'reduce', dev_env)

                ema_eval_metrics = evaluate(train_state.model_ema.module,
                                            train_state.eval_loss,
                                            loader_eval,
                                            services.monitor,
                                            dev_env,
                                            phase_suffix='EMA',
                                            attack=eval_attack)
                eval_metrics = ema_eval_metrics

            if train_state.lr_scheduler is not None:
                # step LR for next epoch
                train_state.lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if services.monitor is not None:
                services.monitor.write_summary(index=epoch,
                                               results=dict(train=train_metrics, eval=eval_metrics))

            if checkpoint_manager is not None:
                # save proper checkpoint with eval metric
                best_checkpoint = checkpoint_manager.save_checkpoint(train_state, eval_metrics)
                best_metric, best_epoch = best_checkpoint.sort_key, best_checkpoint.epoch

            train_state = replace(train_state, epoch=epoch + 1)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    if dev_env.primary and output_dir is not None and output_dir.startswith('gs://'):
        assert checkpoints_dir is not None
        try:
            _logger.info(f"Uploading checkpoints to {output_dir}")
            utils.upload_checkpoints_gcs(checkpoints_dir, output_dir)
            _logger.info(f"Uploaded checkpoints to {output_dir}, removing temporary dir")
            shutil.rmtree(checkpoints_dir)
        except Exception as e:
            _logger.exception(f"Failed to upload checkpoints to GCS: {e}. "
                              "Not removing the temporary dir {checkpoints_dir}.")

    if services.monitor.wandb_run is not None:
        services.monitor.wandb_run.finish()


def setup_train_task(args, dev_env: DeviceEnv, mixup_active: bool):
    with tempfile.TemporaryDirectory() as dst:
        if args.initial_checkpoint is not None and args.initial_checkpoint.startswith("gs://"):
            checkpoint_path = os.path.join(dst, os.path.basename(args.initial_checkpoint))
            tf.io.gfile.copy(args.initial_checkpoint, checkpoint_path)
        else:
            checkpoint_path = args.initial_checkpoint

        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=checkpoint_path)

    if args.finetune is not None:
        # Adapted from https://github.com/facebookresearch/deit/blob/main/main.py#L250
        with tempfile.TemporaryDirectory() as dst:
            if args.finetune.startswith("gs://"):
                checkpoint_path = os.path.join(dst, os.path.basename(args.finetune))
                tf.io.gfile.copy(args.finetune, checkpoint_path)
            else:
                checkpoint_path = args.finetune

            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'fc.weight', 'fc.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        try:
            num_classes = args.num_classes
            model.reset_classifier(num_classes=num_classes)
            print(f"Reset the classifier with {num_classes=}")
        except AttributeError:
            pass

        # interpolate position embedding
        try:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches**0.5)
            # class_token and dist_token are kept unchanged
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens,
                                       size=(new_size, new_size),
                                       mode='bicubic',
                                       align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
        except KeyError:
            # Model has no learned positional embeddings, skipping interpolation
            pass

        model.load_state_dict(checkpoint_model, strict=False)

    if args.finetuning_patch_size is not None:
        assert args.finetuning_patch_size in {2, 4, 8}, "Finetuning patch size can be only 4, 8 or `None`"
        assert isinstance(model, models.xcit.XCiT), "Finetuning patch size is only supported for XCiT"
        _logger.info(f"Adapting patch embedding for finetuning patch size {args.finetuning_patch_size}")
        model.patch_embed.patch_size = args.finetuning_patch_size

        if args.keep_patch_embedding:
            model.patch_embed.proj[0][0].stride = (1, 1)
            if args.finetuning_patch_size == 4:
                model.patch_embed.proj[2][0].stride = (1, 1)
            if args.finetuning_patch_size == 2:
                model.patch_embed.proj[4][0].stride = (1, 1)

        else:
            if args.finetuning_patch_size == 4:
                model.patch_embed.proj = model.patch_embed.proj[-3:]
                model.patch_embed.proj[0] = models.xcit.conv3x3(3, model.embed_dim // 2, 2)

            elif args.finetuning_patch_size == 8:
                if args.reinit_patch_embedding:
                    _logger.info("Re-initializing patch embedding")
                    model.patch_embed = models.xcit.ConvPatchEmbed(args.input_size[-1], 8, 3, model.embed_dim)
                else:
                    model.patch_embed.proj = model.patch_embed.proj[-5:]
                    model.patch_embed.proj[0] = models.xcit.conv3x3(3, model.embed_dim // 4, 2)

    if args.num_classes is None:
        assert hasattr(model,
                       'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if dev_env.primary:
        _logger.info(f'Model {safe_model_name(args.model)} created, '
                     f'param count:{sum([m.numel() for m in model.parameters()])}')

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert args.aug_splits > 1
        model = convert_splitbn_model(model, max(args.aug_splits, 2))

    with tempfile.TemporaryDirectory() as dst:
        if args.resume is not None and args.resume.startswith("gs://"):
            resume_checkpoint_path = os.path.join(dst, os.path.basename(args.resume))
            tf.io.gfile.copy(args.resume, resume_checkpoint_path)
        else:
            resume_checkpoint_path = args.resume

        train_state = setup_model_and_optimizer(
            dev_env=dev_env,
            model=model,
            optimizer=args.opt,
            optimizer_cfg=optimizer_kwargs(cfg=args),
            clip_fn=args.clip_mode if args.clip_grad is not None else None,
            clip_value=args.clip_grad,
            model_ema=args.model_ema,
            model_ema_decay=args.model_ema_decay,
            resume_path=resume_checkpoint_path,
            use_syncbn=args.sync_bn,
            resume_opt=not args.no_resume_opt)

    # setup learning rate schedule and starting epoch
    # FIXME move into updater?
    lr_scheduler, num_epochs = create_scheduler(args, train_state.updater.optimizer)
    if lr_scheduler is not None and train_state.epoch > 0:
        lr_scheduler.step(train_state.epoch)

    # setup loss function
    if args.jsd_loss:
        assert args.aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=args.aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing,
                                               target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    eval_loss_fn = nn.CrossEntropyLoss()

    eps = args.attack_eps / 255
    attack_step_size = args.attack_lr or (1.5 * eps / args.attack_steps)

    if args.adv_training is not None and args.adv_training == "pgd":
        attack_criterion: nn.Module = nn.NLLLoss(reduction="sum")
        train_attack = attacks.make_train_attack(args.attack,
                                                 args.eps_schedule,
                                                 eps,
                                                 args.eps_schedule_period,
                                                 args.zero_eps_epochs,
                                                 attack_step_size,
                                                 args.attack_steps,
                                                 args.attack_norm,
                                                 args.attack_boundaries,
                                                 criterion=attack_criterion,
                                                 num_classes=model.num_classes,
                                                 logits_y=False)
        compute_loss_fn = attacks.AdvTrainingLoss(train_attack, train_loss_fn, eval_mode=not dev_env.type_xla)
    elif args.adv_training is not None and args.adv_training == "trades":
        attack_criterion = nn.KLDivLoss(reduction="sum")
        train_attack = attacks.make_train_attack(args.attack,
                                                 args.eps_schedule,
                                                 eps,
                                                 args.eps_schedule_period,
                                                 args.zero_eps_epochs,
                                                 attack_step_size,
                                                 args.attack_steps,
                                                 args.attack_norm,
                                                 args.attack_boundaries,
                                                 criterion=attack_criterion,
                                                 num_classes=model.num_classes,
                                                 logits_y=True)
        compute_loss_fn = attacks.TRADESLoss(train_attack, train_loss_fn, args.trades_beta)
    else:
        compute_loss_fn = utils.ComputeLossFn(train_loss_fn)

    dev_env.to_device(train_loss_fn, eval_loss_fn, compute_loss_fn)

    if dev_env.primary:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    train_cfg = TrainCfg(
        num_epochs=num_epochs,
        log_interval=args.log_interval,
        recovery_interval=args.recovery_interval,
    )

    train_state = replace(
        train_state,
        lr_scheduler=lr_scheduler,
        train_loss=train_loss_fn,
        eval_loss=eval_loss_fn,
        train_cfg=train_cfg,
    )

    schedule = _SCHEDULES[args.eps_schedule](
        args.attack_eps,
        args.eps_schedule_period,
        args.zero_eps_epochs,
    )

    train_state = utils.AdvTrainState.from_bits(train_state,
                                                compute_loss_fn=compute_loss_fn,
                                                eps_schedule=schedule)

    return train_state


def setup_data(args, default_cfg, dev_env: DeviceEnv, mixup_active: bool):
    data_config = resolve_data_config(vars(args), default_cfg=default_cfg, verbose=dev_env.primary)
    data_config['normalize'] = not (args.no_normalize or args.normalize_model)

    # create the train and eval datasets
    dataset_train = create_dataset(args.dataset,
                                   root=args.data_dir,
                                   split=args.train_split,
                                   is_training=True,
                                   batch_size=args.batch_size,
                                   repeats=args.epoch_repeats)

    dataset_eval = create_dataset(args.dataset,
                                  root=args.data_dir,
                                  split=args.val_split,
                                  is_training=False,
                                  batch_size=args.batch_size)

    # setup mixup / cutmix
    mixup_cfg = None
    if mixup_active:
        mixup_cfg = MixupCfg(prob=args.mixup_prob,
                             switch_prob=args.mixup_switch_prob,
                             mode=args.mixup_mode,
                             mixup_alpha=args.mixup,
                             cutmix_alpha=args.cutmix,
                             cutmix_minmax=args.cutmix_minmax,
                             label_smoothing=args.smoothing,
                             num_classes=args.num_classes)

    # wrap dataset in AugMix helper
    if args.aug_splits > 1:
        if not isinstance(dataset_train, data.IterableDataset):
            dataset_train = AugMixDataset(dataset_train, num_splits=args.aug_splits)
        else:
            dataset_train = IterableAugMixDataset(dataset_train, num_splits=args.aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    if args.no_aug:
        train_aug_cfg = None
    else:
        aa = args.aa if args.aa not in {"None", "none"} else None
        train_aug_cfg = AugCfg(
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            ratio_range=args.ratio,
            scale_range=args.scale,
            hflip_prob=args.hflip,
            vflip_prob=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=aa,
            num_aug_splits=args.aug_splits,
        )

    train_pp_cfg = utils.MyPreprocessCfg(  # type: ignore
        input_size=data_config['input_size'],
        interpolation=train_interpolation,
        crop_pct=data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        aug=train_aug_cfg,
        normalize=data_config['normalize'],
    )

    # if using PyTorch XLA and RandomErasing is enabled, we must normalize and do RE in transforms on CPU
    normalize_in_transform = dev_env.type_xla and args.reprob > 0

    loader_train = create_loader_v2(dataset_train,
                                    batch_size=args.batch_size,
                                    is_training=True,
                                    normalize_in_transform=normalize_in_transform,
                                    pp_cfg=train_pp_cfg,
                                    mix_cfg=mixup_cfg,
                                    num_workers=args.workers,
                                    pin_memory=args.pin_mem,
                                    use_multi_epochs_loader=args.use_multi_epochs_loader,
                                    separate_transform=args.aug_splits > 0)

    if not train_pp_cfg.normalize:
        if normalize_in_transform:
            idx = -2 if args.reprob > 0 else -1
            if args.aug_splits > 0:
                assert isinstance(loader_train.dataset, AugMixDataset)
                assert loader_train.dataset.normalize is not None
                loader_train.dataset.normalize.transforms[idx] = transforms.ToTensor()
            else:
                loader_train.dataset.transform.transforms[idx] = transforms.ToTensor()
        else:
            if args.aug_splits > 0:
                assert isinstance(loader_train.dataset, AugMixDataset)
                assert loader_train.dataset.normalize is not None
                loader_train.dataset.normalize.transforms[-1] = transforms.ToTensor()
            else:
                loader_train.dataset.transform.transforms[-1] = transforms.ToTensor()

            loader_train.mean = None
            loader_train.std = None

    print(f"{train_pp_cfg.normalize = }")
    print(args.reprob > 0 and train_aug_cfg is not None and not train_pp_cfg.normalize)

    print(loader_train.dataset.transform)

    if args.reprob > 0 and train_aug_cfg is not None and not train_pp_cfg.normalize:
        random_erasing = NotNormalizedRandomErasing(probability=train_aug_cfg.re_prob,
                                                    mode=train_aug_cfg.re_mode,
                                                    count=train_aug_cfg.re_count)
        if normalize_in_transform:
            if isinstance(loader_train.dataset, AugMixDataset):
                loader_train.dataset.normalize.transforms[-1] = random_erasing
            else:
                loader_train.dataset.transform.transforms[-1] = random_erasing
        else:
            loader_train.random_erasing = random_erasing

    print(loader_train.dataset.transform)

    eval_pp_cfg = utils.MyPreprocessCfg(  # type: ignore
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        crop_pct=data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        normalize=data_config['normalize'],
    )

    eval_workers = args.workers
    if 'tfds' in args.dataset or 'wds' in args.dataset:
        # FIXME reduces validation padding issues when using TFDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    loader_eval = create_loader_v2(
        dataset_eval,
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        normalize_in_transform=normalize_in_transform,
        pp_cfg=eval_pp_cfg,
        num_workers=eval_workers,
        pin_memory=args.pin_mem,
    )

    if not eval_pp_cfg.normalize:
        loader_eval.dataset.transform.transforms[-1] = transforms.ToTensor()
        loader_eval.mean = None
        loader_eval.std = None

    # Not needed for now
    if args.use_mp_loader and dev_env.type_xla:
        import torch_xla.distributed.parallel_loader as pl
        assert isinstance(loader_train, fetcher.Fetcher)
        assert isinstance(loader_eval, fetcher.Fetcher)
        loader_train.use_mp_loader = True
        loader_train._loader = pl.MpDeviceLoader(loader_train._loader, dev_env.device)
        loader_eval.use_mp_loader = True
        loader_eval._loader = pl.MpDeviceLoader(loader_eval._loader, dev_env.device)

    return data_config, loader_eval, loader_train


def train_one_epoch(
    state: utils.AdvTrainState,
    services: TrainServices,
    loader,
    dev_env: DeviceEnv,
):
    tracker = Tracker()
    # FIXME move loss meter into task specific TaskMetric
    loss_meter = AvgTensor()
    accuracy_meter = AccuracyTopK(topk=(1, ))
    robust_accuracy_meter = AccuracyTopK(topk=(1, ))

    state.model.train()
    state.updater.reset()  # zero-grad

    step_end_idx = len(loader) - 1
    tracker.mark_iter()
    for step_idx, (sample, target) in enumerate(loader):
        tracker.mark_iter_data_end()

        # FIXME move forward + loss into model 'task' wrapper
        with dev_env.autocast():
            loss, output, adv_output = state.compute_loss_fn(state.model, sample, target, state.epoch)

        state.updater.apply(loss)

        tracker.mark_iter_step_end()

        state.updater.after_step(
            after_train_step,
            state,
            services,
            dev_env,
            step_idx,
            step_end_idx,
            tracker,
            loss_meter,
            accuracy_meter,
            robust_accuracy_meter,
            (output, adv_output, target, loss),
        )

        tracker.mark_iter()
        # end for

    if hasattr(state.updater.optimizer, 'sync_lookahead'):
        state.updater.optimizer.sync_lookahead()

    top1, = accuracy_meter.compute().values()
    robust_top1, = robust_accuracy_meter.compute().values()

    return OrderedDict([('loss', loss_meter.compute().item()), ('top1', top1.item()),
                        ('robust_top1', robust_top1.item()), ('eps', state.eps_schedule(state.epoch)),
                        ('lr', state.updater.get_average_lr())])


def after_train_step(
    state: TrainState,
    services: TrainServices,
    dev_env: DeviceEnv,
    step_idx: int,
    step_end_idx: int,
    tracker: Tracker,
    loss_meter: AvgTensor,
    accuracy_meter: AccuracyTopK,
    robust_accuracy_meter: AccuracyTopK,
    tensors: Tuple[torch.Tensor, ...],
):
    """
    After the core loss / backward / gradient apply step, we perform all non-gradient related
    activities here including updating meters, metrics, performing logging, and writing checkpoints.

    Many / most of these operations require tensors to be moved to CPU, they shoud not be done
    every step and for XLA use they should be done via the optimizer step_closure. This function includes
    everything that should be executed within the step closure.

    Args:
        state:
        services:
        dev_env:
        step_idx:
        step_end_idx:
        tracker:
        loss_meter:
        accuracy_meter:
        robust_accuracy_meter:
        tensors:

    Returns:

    """
    end_step = step_idx == step_end_idx

    with torch.no_grad():
        output, adv_output, target, loss = tensors
        loss_meter.update(loss, output.shape[0])

        if len(target.size()) > 1:
            target = target.argmax(dim=-1)

        accuracy_meter.update(output, target)
        if adv_output is not None:
            robust_accuracy_meter.update(adv_output, target)

        if state.model_ema is not None:
            # FIXME should ema update be included here or in train / updater step? does it matter?
            state.model_ema.update(state.model)

        state = replace(state, step_count_global=state.step_count_global + 1)
        cfg = state.train_cfg

        if services.monitor is not None and end_step or (step_idx + 1) % cfg.log_interval == 0:
            global_batch_size = dev_env.world_size * output.shape[0]
            loss_avg = loss_meter.compute()
            top1, = accuracy_meter.compute().values()
            robust_top1, = robust_accuracy_meter.compute().values()

            if services.monitor is not None:
                lr_avg = state.updater.get_average_lr()
                services.monitor.log_step('Train',
                                          step_idx=step_idx,
                                          step_end_idx=step_end_idx,
                                          epoch=state.epoch,
                                          loss=loss_avg.item(),
                                          top1=top1.item(),
                                          robust_top1=robust_top1.item(),
                                          rate=tracker.get_avg_iter_rate(global_batch_size),
                                          lr=lr_avg)

        if services.checkpoint is not None and cfg.recovery_interval and (end_step or (step_idx + 1) %
                                                                          cfg.recovery_interval == 0):
            services.checkpoint.save_recovery(state)

        if state.lr_scheduler is not None:
            # FIXME perform scheduler update here or via updater after_step call?
            state.lr_scheduler.step_update(num_updates=state.step_count_global)


def evaluate(model: nn.Module,
             loss_fn: nn.Module,
             loader,
             logger: Monitor,
             dev_env: DeviceEnv,
             phase_suffix: str = '',
             log_interval: int = 10,
             attack: Optional[AttackFn] = None):
    tracker = Tracker()
    losses_m = AvgTensor()
    # FIXME move loss and accuracy modules into task specific TaskMetric obj
    accuracy_m = AccuracyTopK()
    robust_accuracy_m = AccuracyTopK()

    model.eval()

    end_idx = len(loader) - 1
    tracker.mark_iter()
    with torch.no_grad():
        for step_idx, (sample, target) in enumerate(loader):
            tracker.mark_iter_data_end()
            last_step = step_idx == end_idx

            with dev_env.autocast():
                output = model(sample)
                loss = loss_fn(output, target)

                if attack is not None:
                    with torch.enable_grad():
                        if dev_env.type_xla:
                            model.train()
                        adv_sample = attack(model, sample, target)
                        model.eval()
                        adv_output = model(adv_sample)
                else:
                    adv_output = None

            # FIXME, explictly marking step for XLA use since I'm not using the parallel xm loader
            # need to investigate whether parallel loader wrapper is helpful on tpu-vm or
            # only use for 2-vm setup.
            if dev_env.type_xla:
                dev_env.mark_step()
            elif dev_env.type_cuda:
                dev_env.synchronize()

            # FIXME uncommenting this fixes race btw model `output`/`loss` and loss_m/accuracy_m meter input
            # for PyTorch XLA GPU use.
            # This issue does not exist for normal PyTorch w/ GPU (CUDA) or PyTorch XLA w/ TPU.
            # loss.item()

            tracker.mark_iter_step_end()
            losses_m.update(loss, output.size(0))
            accuracy_m.update(output, target)

            if adv_output is not None:
                robust_accuracy_m.update(adv_output, target)

            if last_step or step_idx % log_interval == 0:
                top1, top5 = accuracy_m.compute().values()
                if adv_output is not None:
                    robust_top1, _ = robust_accuracy_m.compute().values()
                else:
                    robust_top1 = None

                loss_avg = losses_m.compute()
                logger.log_step(
                    'Eval',
                    step_idx=step_idx,
                    step_end_idx=end_idx,
                    loss=loss_avg.item(),
                    top1=top1.item(),
                    top5=top5.item(),
                    robust_top1=robust_top1.item() if robust_top1 is not None else None,
                    phase_suffix=phase_suffix,
                )
            tracker.mark_iter()

    top1, top5 = accuracy_m.compute().values()
    robust_top1, _ = robust_accuracy_m.compute().values()
    results = OrderedDict([
        ('loss', losses_m.compute().item()),
        ('top1', top1.item()),
        ('robust_top1', robust_top1.item()),
    ])
    return results


def _mp_entry(*args):
    main()


if __name__ == '__main__':
    main()

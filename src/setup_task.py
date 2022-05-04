import logging
import os
import tempfile
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Tuple

import tensorflow as tf
import torch
import torch.nn as nn
import torch.utils.data as data
from timm.bits import CheckpointManager, DeviceEnv, TrainCfg, TrainState, setup_model_and_optimizer
from timm.data import AugCfg, AugMixDataset, MixupCfg, create_loader_v2, fetcher, resolve_data_config
from timm.data.dataset_factory import create_dataset
from timm.loss import (BinaryCrossEntropy, JsdCrossEntropy, LabelSmoothingCrossEntropy,
                       SoftTargetCrossEntropy)
from timm.models import convert_splitbn_model, create_model, safe_model_name
from timm.optim import optimizer_kwargs
from timm.utils.model_ema import ModelEmaV2
from timm.scheduler import create_scheduler
from torchvision import transforms

from src.attacks import _SCHEDULES, AttackCfg
from src.iterable_augmix_dataset import IterableAugMixDataset
from src.random_erasing import NotNormalizedRandomErasing

from . import (  # Models import needed to register the extra models that are not in timm
    attacks, utils)

_logger = logging.getLogger('train')


def setup_data(args, default_cfg, dev_env: DeviceEnv, mixup_active: bool):
    data_config = resolve_data_config(vars(args), default_cfg=default_cfg, verbose=dev_env.primary)
    data_config['normalize'] = not (args.no_normalize or args.normalize_model)

    if args.combine_dataset is not None:
        train_combine_batch_size = int(args.batch_size * args.combined_dataset_ratio)
        train_batch_size = args.batch_size - train_combine_batch_size
    else:
        train_combine_batch_size = 0  # This is not used in practice
        train_batch_size = args.batch_size

    # create the train and eval datasets
    dataset_train = create_dataset(args.dataset,
                                   root=args.data_dir,
                                   split=args.train_split,
                                   is_training=True,
                                   batch_size=train_batch_size,
                                   repeats=args.epoch_repeats)

    if args.combine_dataset is not None:
        data_dir = args.combine_data_dir or args.data_dir
        dataset_train_combine = create_dataset(args.combine_dataset,
                                               root=data_dir,
                                               split=args.train_split,
                                               is_training=True,
                                               batch_size=train_combine_batch_size,
                                               repeats=args.epoch_repeats)
    else:
        dataset_train_combine = None

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
            if dataset_train_combine is not None:
                dataset_train_combine = AugMixDataset(dataset_train_combine, num_splits=args.aug_splits)
        else:
            dataset_train = IterableAugMixDataset(dataset_train, num_splits=args.aug_splits)
            if dataset_train_combine is not None:
                dataset_train_combine = IterableAugMixDataset(dataset_train_combine,
                                                              num_splits=args.aug_splits)

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
                                    batch_size=train_batch_size,
                                    is_training=True,
                                    normalize_in_transform=normalize_in_transform,
                                    pp_cfg=train_pp_cfg,
                                    mix_cfg=mixup_cfg,
                                    num_workers=args.workers,
                                    pin_memory=args.pin_mem,
                                    use_multi_epochs_loader=args.use_multi_epochs_loader,
                                    separate_transform=args.aug_splits > 0)

    if dataset_train_combine is not None:
        loader_train_combine = create_loader_v2(dataset_train_combine,
                                                batch_size=train_combine_batch_size,
                                                is_training=True,
                                                normalize_in_transform=normalize_in_transform,
                                                pp_cfg=train_pp_cfg,
                                                mix_cfg=mixup_cfg,
                                                num_workers=args.workers,
                                                pin_memory=args.pin_mem,
                                                use_multi_epochs_loader=args.use_multi_epochs_loader,
                                                separate_transform=args.aug_splits > 0)
    else:
        loader_train_combine = None

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
        if loader_train_combine is not None:
            if normalize_in_transform:
                idx = -2 if args.reprob > 0 else -1
                if args.aug_splits > 0:
                    assert isinstance(loader_train_combine.dataset, AugMixDataset)
                    assert loader_train_combine.dataset.normalize is not None
                    loader_train_combine.dataset.normalize.transforms[idx] = transforms.ToTensor()
                else:
                    loader_train_combine.dataset.transform.transforms[idx] = transforms.ToTensor()
            else:
                if args.aug_splits > 0:
                    assert isinstance(loader_train_combine.dataset, AugMixDataset)
                    assert loader_train_combine.dataset.normalize is not None
                    loader_train_combine.dataset.normalize.transforms[-1] = transforms.ToTensor()
                else:
                    loader_train_combine.dataset.transform.transforms[-1] = transforms.ToTensor()

                loader_train_combine.mean = None
                loader_train_combine.std = None

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

        if loader_train_combine is not None:
            if normalize_in_transform:
                if isinstance(loader_train_combine.dataset, AugMixDataset):
                    loader_train_combine.dataset.normalize.transforms[-1] = random_erasing
                else:
                    loader_train_combine.dataset.transform.transforms[-1] = random_erasing
            else:
                loader_train_combine.random_erasing = random_erasing

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
        if loader_train_combine is not None:
            assert isinstance(loader_train_combine, fetcher.Fetcher)
            loader_train_combine.use_mp_loader = True
            loader_train_combine._loader = pl.MpDeviceLoader(loader_train._loader, dev_env.device)

    if loader_train_combine is not None:
        loader_train = utils.CombinedLoaders(loader_train, loader_train_combine)

    return data_config, loader_eval, loader_train


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

        # FIXME: probably not needed as we call `reset_classifier` below
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'fc.weight', 'fc.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                _logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        try:
            num_classes = args.num_classes
            model.reset_classifier(num_classes=num_classes)
            _logger.info(f"Reset the classifier with {num_classes=}")
        except AttributeError:
            _logger.warn("Could not reset classifier on model")

        # interpolate position embedding
        # FIXME: move to a function to clean up
        try:
            checkpoint_model['pos_embed'] = utils.interpolate_position_embeddings(model, checkpoint_model)
        except KeyError:
            # Model has no learned positional embeddings, skipping interpolation
            pass

        model.load_state_dict(checkpoint_model, strict=False)

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

    if args.adv_training is not None:
        attack_cfg = resolve_attack_cfg(args)
        if args.adv_training == "pgd":
            compute_loss_fn = attacks.AdvTrainingLoss(attack_cfg,
                                                      train_loss_fn,
                                                      dev_env,
                                                      model.num_classes,
                                                      eval_mode=not dev_env.type_xla)
        elif args.adv_training == "trades":
            compute_loss_fn = attacks.TRADESLoss(attack_cfg,
                                                 train_loss_fn,
                                                 args.trades_beta,
                                                 dev_env,
                                                 model.num_classes,
                                                 eval_mode=not dev_env.type_xla)
        else:
            raise ValueError("Adversarial training mode not supported")
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


def update_state_with_norm_model(dev_env: DeviceEnv, train_state: TrainState,
                                 data_config: Dict[str, Any]) -> utils.AdvTrainState:
    train_state = replace(train_state,
                          model=utils.normalize_model(train_state.model,
                                                      mean=data_config["mean"],
                                                      std=data_config["std"]))
    train_state = replace(train_state, model=dev_env.to_device(train_state.model))

    if train_state.model_ema is not None:
        assert isinstance(train_state.model_ema, ModelEmaV2)
        new_model_ema = ModelEmaV2(train_state.model, decay=train_state.model_ema.decay)
        train_state = replace(train_state, model_ema=dev_env.to_device(new_model_ema))

    return train_state


def setup_checkpoints_output(args: Dict[str, Any], args_text: str, data_config: Dict[str, Any],
                             eval_metric: str) -> Tuple[CheckpointManager, str, str]:
    if args["experiment"]:
        exp_name = args["experiment"]
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args["model"]),
            str(data_config['input_size'][-1])
        ])

    output_dir = utils.get_outdir(args["output"] if args["output"] else './output/train', exp_name, inc=True)
    if output_dir.startswith("gs://"):
        checkpoints_dir = utils.get_outdir('./output/tmp/', exp_name, inc=True)
        _logger.info(f"Temporarily saving checkpoints in {checkpoints_dir}")
    else:
        checkpoints_dir = output_dir

    checkpoint_manager = CheckpointManager(hparams=args,
                                           checkpoint_dir=checkpoints_dir,
                                           recovery_dir=output_dir,
                                           metric_name=eval_metric,
                                           metric_decreasing=True if eval_metric == 'loss' else False,
                                           max_history=args["checkpoint_hist"])

    if output_dir.startswith("gs://"):
        with tf.io.gfile.GFile(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    else:
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    return checkpoint_manager, output_dir, checkpoints_dir


def resolve_attack_cfg(args, eval=False) -> AttackCfg:
    if eval:
        # Make train targeted attack untargeted
        name = args.attack.split("targeted_")[-1]
        eps = (args.eval_attack_eps or args.attack_eps) / 255
    else:
        name = args.attack
        eps = args.attack_eps / 255
    step_size = args.attack_lr or (1.5 * eps / args.attack_steps)

    return AttackCfg(name=name,
                     eps=eps,
                     eps_schedule=args.eps_schedule,
                     eps_schedule_period=args.eps_schedule_period,
                     zero_eps_epochs=args.zero_eps_epochs,
                     step_size=step_size,
                     steps=args.attack_steps,
                     norm=args.attack_norm,
                     boundaries=args.attack_boundaries)

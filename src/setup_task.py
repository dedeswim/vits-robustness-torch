import os
import tempfile
from dataclasses import replace

import tensorflow as tf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from timm.bits import DeviceEnv, TrainCfg, setup_model_and_optimizer
from timm.data import (AugCfg, AugMixDataset, MixupCfg, create_loader_v2, fetcher, resolve_data_config)
from timm.data.dataset_factory import create_dataset
from timm.loss import (BinaryCrossEntropy, JsdCrossEntropy, LabelSmoothingCrossEntropy,
                       SoftTargetCrossEntropy)
from timm.models import convert_splitbn_model, create_model, safe_model_name
from timm.optim import optimizer_kwargs
from timm.scheduler import create_scheduler
from torchvision import transforms

from src.attacks import _SCHEDULES
from src.iterable_augmix_dataset import IterableAugMixDataset
from src.random_erasing import NotNormalizedRandomErasing

from . import (  # Models import needed to register the extra models that are not in timm
    attacks, utils)


def setup_data(args, default_cfg, dev_env: DeviceEnv, mixup_active: bool):
    if args.data_dir.startswith("gs://"):
        utils.check_bucket_zone(args.data_dir, "large-ds")

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
        assert isinstance(model, timm.models.xcit.XCiT), "Finetuning patch size is only supported for XCiT"
        _logger.info(f"Adapting patch embedding for finetuning patch size {args.finetuning_patch_size}")
        model.patch_embed.patch_size = args.finetuning_patch_size

        # FIXME use models from `models` module
        if args.keep_patch_embedding:
            model.patch_embed.proj[0][0].stride = (1, 1)
            if args.finetuning_patch_size == 4:
                model.patch_embed.proj[2][0].stride = (1, 1)
            if args.finetuning_patch_size == 2:
                model.patch_embed.proj[4][0].stride = (1, 1)

        # FIXME: remove this?
        else:
            if args.finetuning_patch_size == 4:
                model.patch_embed.proj = model.patch_embed.proj[-3:]
                model.patch_embed.proj[0] = timm.models.xcit.conv3x3(3, model.embed_dim // 2, 2)

            elif args.finetuning_patch_size == 8:
                if args.reinit_patch_embedding:
                    _logger.info("Re-initializing patch embedding")
                    model.patch_embed = timm.models.xcit.ConvPatchEmbed(args.input_size[-1], 8, 3,
                                                                        model.embed_dim)
                else:
                    model.patch_embed.proj = model.patch_embed.proj[-5:]
                    model.patch_embed.proj[0] = timm.models.xcit.conv3x3(3, model.embed_dim // 4, 2)

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
        # FIXME: move this inside of AdvTrainingLoss constructor
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
                                                 logits_y=False,
                                                 dev_env=dev_env)
        compute_loss_fn = attacks.AdvTrainingLoss(train_attack, train_loss_fn, eval_mode=not dev_env.type_xla)
    elif args.adv_training is not None and args.adv_training == "trades":
        # FIXME: move this inside of TRADESLoss constructor
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
                                                 logits_y=True,
                                                 dev_env=dev_env)
        compute_loss_fn = attacks.TRADESLoss(train_attack,
                                             train_loss_fn,
                                             args.trades_beta,
                                             eval_mode=not dev_env.type_xla)
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

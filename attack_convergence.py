import logging
from multiprocessing.sharedctypes import Value
from pathlib import Path

import numpy as np
import timm
from timm.bits import initialize_device
from timm.data import create_dataset, create_loader_v2, resolve_data_config
from timm.models import apply_test_time_pool
from timm.utils import setup_default_logging
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from zmq import device

from src import attacks, utils
from src.random import random_seed
from validate import parser

_logger = logging.getLogger('validate')

parser.add_argument('--runs', type=int, default=20, metavar='N', help='The number of runs')
parser.add_argument('--n-points', type=int, default=100, metavar='N', help='The number of points')
parser.add_argument('--output-file', type=str, default=None, metavar='N', help='The output file')
parser.add_argument('--steps-to-try',
                    type=int,
                    nargs="+",
                    default=(0, 1, 2, 5, 10, 50, 100, 200, 500),
                    metavar='X Y Z',
                    help='The number of steps to try')
parser.add_argument('--one-instance',
                    action='store_true',
                    help='Run only one instance and save the losses at each step')


def main():
    setup_default_logging()
    args = parser.parse_args()

    dev_env = initialize_device(force_cpu=args.force_cpu, amp=args.amp)
    random_seed(args.seed, dev_env.global_rank)

    if args.output_file is None:
        args.output_file = f"{args.model}.csv"
    output_path = Path(args.output_file)
    csv_writer = utils.GCSSummaryCsv(output_path.parent, filename=output_path.name)

    model = timm.create_model(args.model, pretrained=not args.checkpoint, checkpoint_path=args.checkpoint)
    model = dev_env.to_device(model)
    model.eval()

    criterion = dev_env.to_device(nn.CrossEntropyLoss(reduction='none'))

    eps = args.attack_eps / 255
    lr = args.attack_lr or (1.5 * eps / args.attack_steps)
    attack_criterion = nn.NLLLoss(reduction="sum")

    dataset = create_dataset(root=args.data,
                             name=args.dataset,
                             split=args.split,
                             download=args.dataset_download)

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    data_config['normalize'] = not (args.no_normalize or args.normalize_model)

    if args.normalize_model:
        mean = args.mean or data_config["mean"]
        std = args.std or data_config["std"]
        model = utils.normalize_model(model, mean=mean, std=std)

    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)
    eval_pp_cfg = utils.MyPreprocessCfg(  # type: ignore
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        crop_pct=1.0 if test_time_pool else data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        normalize=data_config['normalize'],
    )

    if args.one_instance:
        args.steps_to_try = [max(args.steps_to_try)]

    if args.n_points % args.batch_size != 0:
        raise ValueError(f"n_points ({args.n_points}) must be a multiple of batch_size ({args.batch_size})")

    loader = create_loader_v2(dataset,
                              batch_size=args.batch_size,
                              is_training=False,
                              pp_cfg=eval_pp_cfg,
                              num_workers=args.workers,
                              pin_memory=args.pin_mem)
    if not eval_pp_cfg.normalize:
        loader.dataset.transform.transforms[-1] = transforms.ToTensor()

    correctly_classified_samples = []
    correctly_classified_targets = []
    correctly_classified_ids = []

    _logger.info("Starting creation of correctly classified DataSet and DataLoader")

    for batch_idx, (sample, target) in enumerate(loader):
        predicted_classes = model(sample).argmax(-1)
        accuracy_mask = predicted_classes.eq(target)
        print(f"Batch {batch_idx} accuracy: {accuracy_mask.sum() / sample.shape[0]}")
        # Get correctly classified samples, targets, and ids
        batch_correctly_classified_samples = sample[accuracy_mask]
        batch_correctly_classified_targets = target[accuracy_mask]
        batch_correctly_classified_ids = accuracy_mask.nonzero().flatten() + batch_idx * args.batch_size

        correctly_classified_samples.append(dev_env.to_cpu(batch_correctly_classified_samples))
        correctly_classified_targets.append(dev_env.to_cpu(batch_correctly_classified_targets))
        correctly_classified_ids.append(dev_env.to_cpu(batch_correctly_classified_ids))

        if len(torch.cat(correctly_classified_samples)) >= args.n_points:
            correctly_classified_samples = torch.cat(correctly_classified_samples)[:args.n_points]
            correctly_classified_targets = torch.cat(correctly_classified_targets)[:args.n_points]
            correctly_classified_ids = torch.cat(correctly_classified_ids)[:args.n_points]
            break

    if len(correctly_classified_samples) != args.n_points:
        raise ValueError("Impossible to have enough correctly classified samples.")

    correctly_classified_dataset = TensorDataset(correctly_classified_samples, correctly_classified_targets,
                                                 correctly_classified_ids)
    experiment_batch_size = 1 if args.one_instance else args.batch_size
    correctly_classified_loader = DataLoader(correctly_classified_dataset,
                                             batch_size=experiment_batch_size)

    _logger.info("Created correctly classified DataSet and DataLoader")

    for batch_idx, (sample, target, sample_id) in zip(range(args.n_points // experiment_batch_size),
                                                      correctly_classified_loader):
        sample, target, sample_id = dev_env.to_device(sample, target, sample_id)
        for run in range(args.runs):
            for step in args.steps_to_try:
                random_seed(run, dev_env.global_rank)
                attack = attacks.make_attack(args.attack,
                                             eps,
                                             lr,
                                             step,
                                             args.attack_norm,
                                             args.attack_boundaries,
                                             attack_criterion,
                                             dev_env=dev_env,
                                             return_losses=True)
                _logger.info(
                    f"Points ({batch_idx * experiment_batch_size}, {batch_idx * experiment_batch_size + experiment_batch_size}) - run {run} - steps {step}"
                )
                # Make sure that all samples are correctly classified (only need to check the first time)
                if run == 0:
                    logits = model(sample)
                    assert dev_env.to_cpu(logits.argmax(-1).eq(target).all()).item()

                if dev_env.type_xla:
                    # Change model to `train` if on XLA, and backup batchnorm stats
                    batch_stats_backup = utils.backup_batchnorm_stats(model)
                    model.train()
                else:
                    batch_stats_backup  = None
                
                # Attack sample
                adv_sample, intermediate_losses = attack(model, sample, target)
                final_losses = criterion(model(adv_sample), target)
                
                if dev_env.type_xla:
                    # Change model back to `eval` if on XLA, and restore batchnorm stats
                    assert batch_stats_backup is not None
                    utils.restore_batchnorm_stats(model, batch_stats_backup)
                    model.eval()

                final_losses_np = dev_env.to_cpu(final_losses).detach().numpy()
                sample_id_numpy = dev_env.to_cpu(sample_id).detach().numpy()
                if not args.one_instance:
                    for point_id, loss in zip(sample_id_numpy, final_losses_np):
                        row_to_write = {"point": point_id, "seed": run, "steps": step, "loss": loss}
                        csv_writer.update(row_to_write)
                        _logger.info(f"Point {point_id} - run {run} - steps {step} - loss: {loss:.4f}")
                else:
                    intermediate_losses_np = np.concatenate(
                        [dev_env.to_cpu(intermediate_losses).detach().numpy(), final_losses_np])

                    for step_idx, loss in enumerate(intermediate_losses_np):
                        row_to_write = {
                            "point": sample_id_numpy.item(),
                            "seed": run,
                            "steps": step_idx,
                            "loss": loss
                        }
                        csv_writer.update(row_to_write)


def _mp_entry(*args):
    main()


if __name__ == '__main__':
    main()

import logging
from multiprocessing.sharedctypes import Value
from pathlib import Path

import timm
from timm.bits import initialize_device
from timm.data import create_dataset, create_loader_v2, resolve_data_config
from timm.models import apply_test_time_pool
from timm.utils import setup_default_logging
from torch import nn
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

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
        args.batch_size = 1

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

    _logger.info("Starting creation of correctly classified DataSet and DataLoader")

    for (sample, target) in loader:
        predicted_classes = model(sample).argmax(-1)
        batch_correctly_classified_samples = sample[predicted_classes == target]
        batch_correctly_classified_targets = target[predicted_classes == target]
        correctly_classified_samples.append(batch_correctly_classified_samples)
        correctly_classified_targets.append(batch_correctly_classified_targets)
        if len(correctly_classified_samples) >= args.n_points:
            correctly_classified_samples = torch.cat(correctly_classified_samples)[:args.n_points]
            correctly_classified_targets = torch.cat(correctly_classified_targets)[:args.n_points]
            break

    if len(correctly_classified_samples) != args.n_points:
        raise ValueError("Impossible to have enough correctly classified samples.")
    
    correctly_classified_dataset = TensorDataset(correctly_classified_samples, correctly_classified_targets)
    correctly_classified_loader = create_loader_v2(correctly_classified_dataset,
                                                   batch_size=args.batch_size,
                                                   is_training=False,
                                                   pp_cfg=eval_pp_cfg,
                                                   num_workers=args.workers,
                                                   pin_memory=args.pin_mem)

    _logger.info("Created correctly classified DataSet and DataLoader")

    for batch_idx, (sample, target) in zip(range(args.n_points // args.batch_size), correctly_classified_loader):
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
                    f"Points ({batch_idx * args.batch_size}, {batch_idx * args.batch_size + args.batch_size}) - run {run} - steps {step}"
                )
                adv_sample, intermediate_losses = attack(model, sample, target)
                final_losses = criterion(model(adv_sample), target)
                final_losses_np = dev_env.to_cpu(final_losses).detach().numpy()
                intermediate_losses.append(final_losses)
                if not args.one_instance:
                    for point_idx, loss in enumerate(final_losses_np):
                        point = batch_idx * args.batch_size + point_idx
                        row_to_write = {"point": point, "seed": run, "steps": step, "loss": loss}
                        csv_writer.update(row_to_write)
                        _logger.info(f"Point {point_idx} - run {run} - steps {step} - loss: {loss:.4f}")
                else:
                    intermediate_losses_np = dev_env.to_cpu(intermediate_losses).detach().numpy()
                    for step_idx, loss in enumerate(intermediate_losses_np):
                        row_to_write = {"point": batch_idx, "seed": run, "steps": step_idx, "loss": loss}
                        csv_writer.update(row_to_write)


def _mp_entry(*args):
    main()


if __name__ == '__main__':
    main()

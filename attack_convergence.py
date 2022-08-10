import logging
from pathlib import Path

import timm
from timm.bits import initialize_device
from timm.data import create_dataset, create_loader_v2, resolve_data_config
from timm.models import apply_test_time_pool
from timm.utils import setup_default_logging
from torch import nn
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
                    default=(1, 2, 5, 10, 50, 100, 200, 500),
                    metavar='X Y Z',
                    help='The number of steps to try')


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
    loader = create_loader_v2(dataset,
                              batch_size=args.batch_size,
                              is_training=False,
                              pp_cfg=eval_pp_cfg,
                              num_workers=args.workers,
                              pin_memory=args.pin_mem)
    if not eval_pp_cfg.normalize:
        loader.dataset.transform.transforms[-1] = transforms.ToTensor()

    if args.n_points % args.batch_size != 0:
        raise ValueError(f"n_points ({args.n_points}) must be a multiple of batch_size ({args.batch_size})")

    for batch_idx, (sample, target) in zip(range(args.n_points // args.batch_size), loader):
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
                                             dev_env=dev_env)
                _logger.info(f"Point {batch_idx} - run {run} - steps {step}")
                adv_sample = attack(model, sample, target)
                losses = criterion(model(adv_sample), target)
                losses_np = dev_env.to_cpu(losses.detach().numpy())
                for point_idx, loss in enumerate(losses_np):
                    point = batch_idx * args.batch_size + point_idx
                    row_to_write = {"point": point, "seed": run, "steps": step, "loss": loss}
                    csv_writer.update(row_to_write)
                    _logger.info(f"Point {point} - run {run} - steps {step} - loss: {loss:.4f}")


def _mp_entry(*args):
    main()


if __name__ == '__main__':
    main()

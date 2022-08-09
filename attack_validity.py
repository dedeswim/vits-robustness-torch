import logging

import numpy as np
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


def main():
    setup_default_logging()
    args = parser.parse_args()

    dev_env = initialize_device(force_cpu=args.force_cpu, amp=args.amp)
    random_seed(args.seed, dev_env.global_rank)

    if args.output_file is None:
        args.output_file = f"{args.model}.npy"

    model = timm.create_model(args.model, pretrained=True)
    model = dev_env.to_device(model)

    eps = args.attack_eps / 255
    lr = args.attack_lr or (1.5 * eps / args.attack_steps)
    attack_criterion = nn.NLLLoss(reduction="sum")
    attack = attacks.make_attack(args.attack,
                                 eps,
                                 lr,
                                 args.attack_steps,
                                 args.attack_norm,
                                 args.attack_boundaries,
                                 attack_criterion,
                                 dev_env=dev_env,
                                 return_losses=True)

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
                              batch_size=1,
                              is_training=False,
                              pp_cfg=eval_pp_cfg,
                              num_workers=args.workers,
                              pin_memory=args.pin_mem)
    if not eval_pp_cfg.normalize:
        loader.dataset.transform.transforms[-1] = transforms.ToTensor()

    all_losses = [[] for _ in range(args.n_points)]

    for point, (sample, target) in zip(range(args.n_points), loader):
        for run in range(args.runs):
            _logger.info(f"Point {point} - run {run}")
            _, losses = attack(model, sample, target)
            all_losses[point].append(losses)

    all_losses_array = np.array(all_losses)
    np.save(args.output_file, all_losses_array)


def _mp_entry(*args):
    main()


if __name__ == '__main__':
    main()

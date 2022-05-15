import gc
import logging
from pathlib import Path

from timm.data import create_dataset, create_loader_v2
from timm.bits import initialize_device
from timm.utils import setup_default_logging

from src import utils
from src.utils import GCSSummaryCsv
from validate import parser, validate

_logger = logging.getLogger('validate')

parser.add_argument('--steps-to-try',
                    type=int,
                    nargs='+',
                    default=(1, 2, 10, 100),
                    metavar='X Y Z',
                    help='The attack steps to try')
parser.add_argument('--run-apgd-ce', action='store_true', default=False, help='Run also APGD-CE')
parser.add_argument('--checkpoints-dir',
                    type=str,
                    default='',
                    metavar='DIR',
                    help='The directory containing the checkpoints')
parser.add_argument('--epochs-to-try',
                    type=int,
                    nargs='+',
                    default=[0] + list(range(9, 109, 10)),
                    metavar='X Y Z',
                    help='The attack steps to try')


def validate_epoch(args, checkpoints_dir: Path, epoch: int, steps_to_try: int, run_apgd_ce: bool,
                   csv_writer: GCSSummaryCsv, dev_env, dataset, loader):
    args.checkpoint = checkpoints_dir + f"/checkpoint-{epoch}.pth.tar"
    model = utils.load_model_from_gcs(args.checkpoint,
                                      args.model,
                                      pretrained=args.pretrained,
                                      num_classes=args.num_classes,
                                      in_chans=3,
                                      global_pool=args.gp,
                                      scriptable=args.torchscript)

    for attack_steps in steps_to_try:
        args.attack = "pgd"
        _logger.info(f"Starting validation with PGD-{attack_steps} at epoch {epoch}")
        args.attack_steps = attack_steps
        results = validate(args, dev_env, dataset, model, loader)
        results["attack"] = "pgd"
        results["attack_steps"] = attack_steps
        results["model"] = args.model
        results["epoch"] = epoch
        if dev_env.primary:
            csv_writer.update(results)
        gc.collect()

    if run_apgd_ce:
        args.attack = "apgd-ce"
        _logger.info(f"Starting validation with APGD-CE at epoch {epoch}")
        results = validate(args, dev_env, dataset, model, loader)
        results["attack"] = "apgd-ce"
        results["attack_steps"] = None
        results["model"] = args.model
        results["epoch"] = epoch
        if dev_env.primary:
            csv_writer.update(results)
        gc.collect()


def main():
    setup_default_logging()
    args = parser.parse_args()
    checkpoints_dir = args.checkpoints_dir
    run_apgd_ce = args.run_apgd_ce
    steps_to_try = args.steps_to_try
    csv_writer = GCSSummaryCsv(checkpoints_dir)

    dev_env = initialize_device()
    dataset = create_dataset(root=args.data,
                             name=args.dataset,
                             split=args.split,
                             download=args.dataset_download,
                             load_bytes=args.tf_preprocessing,
                             class_map=args.class_map)

    for epoch in args.epochs_to_try:
        validate_epoch(args, checkpoints_dir, epoch, steps_to_try, run_apgd_ce, csv_writer, dev_env, dataset)


def _mp_entry(*args):
    main()


if __name__ == "__main__":
    main()

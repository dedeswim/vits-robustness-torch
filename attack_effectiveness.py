import logging
from pathlib import Path

from timm.utils import setup_default_logging

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
                    default=[0] + list(range(9, 99, 10)),
                    metavar='X Y Z',
                    help='The attack steps to try')


def validate_epoch(args, checkpoints_dir: Path, epoch: int, steps_to_try: int, run_apgd_ce: bool,
                   csv_writer: GCSSummaryCsv):
    args.checkpoint = checkpoints_dir + f"/checkpoint-{epoch}.pth.tar"
    for attack_steps in steps_to_try:
        args.attack = "pgd"
        _logger.info(f"Starting validation with PGD-{attack_steps}")
        args.attack_steps = attack_steps
        results = validate(args)
        results["attack"] = "pgd"
        results["attack_steps"] = attack_steps
        results["model"] = args.model
        csv_writer.update(results)

    if run_apgd_ce:
        args.attack = "apgd-ce"
        _logger.info(f"Starting validation with APGD-CE")
        results = validate(args)
        results["attack"] = "apgd-ce"
        results["attack_steps"] = None
        results["model"] = args.model
        csv_writer.update(results)


def main():
    setup_default_logging()
    args = parser.parse_args()
    checkpoints_dir = args.checkpoints_dir
    run_apgd_ce = args.run_apgd_ce
    steps_to_try = args.steps_to_try
    csv_writer = GCSSummaryCsv(checkpoints_dir)
    for epoch in args.epochs_to_try:
        validate_epoch(args, checkpoints_dir, epoch, steps_to_try, run_apgd_ce, csv_writer)


def _mp_entry(*args):
    main()


if __name__ == "__main__":
    main()

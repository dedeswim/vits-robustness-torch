from src import setup_task
from src.arg_parser import parse_args


def test_resolve_attack_cfg():
    args, _ = parse_args()
    args.attack_eps = 4
    args.attack_steps = 10
    attack_cfg = setup_task.resolve_attack_cfg(args)

    eps = 4 / 255
    assert attack_cfg.eps == eps

    expected_step_size = 1.5 * eps / args.attack_steps
    assert attack_cfg.step_size == expected_step_size


def test_resolve_attack_cfg_attack_lr():
    args, _ = parse_args()
    args.attack_eps = 4
    args.attack_steps = 10
    args.attack_lr = 15

    attack_cfg = setup_task.resolve_attack_cfg(args)
    assert attack_cfg.step_size == args.attack_lr


def test_resolve_attack_cfg_eval_eps():
    args, _ = parse_args()
    args.eval_attack_eps = 8
    args.attack_steps = 10
    attack_cfg = setup_task.resolve_attack_cfg(args, eval=True)

    eps = 8 / 255
    assert attack_cfg.eps == eps

    expected_step_size = 1.5 * eps / args.attack_steps
    assert attack_cfg.step_size == expected_step_size


def test_resolve_attack_cfg_eval_name():
    args, _ = parse_args()
    args.attack = "targeted_pgd"
    attack_cfg = setup_task.resolve_attack_cfg(args, eval=True)

    assert attack_cfg.name == "pgd"

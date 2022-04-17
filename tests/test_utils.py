import pytest
from src import setup_task
from src.utils import adapt_model_patches
from src.arg_parser import parse_args

from timm.models import xcit


def test_resolve_attack_cfg():
    args, _ = parse_args(additional_args=["foo"])
    args.attack_eps = 4
    args.attack_steps = 10
    attack_cfg = setup_task.resolve_attack_cfg(args)

    eps = 4 / 255
    assert attack_cfg.eps == eps

    expected_step_size = 1.5 * eps / args.attack_steps
    assert attack_cfg.step_size == expected_step_size


def test_resolve_attack_cfg_attack_lr():
    args, _ = parse_args(additional_args=["foo"])
    args.attack_eps = 4
    args.attack_steps = 10
    args.attack_lr = 15

    attack_cfg = setup_task.resolve_attack_cfg(args)
    assert attack_cfg.step_size == args.attack_lr


def test_resolve_attack_cfg_eval_eps():
    args, _ = parse_args(additional_args=["foo"])
    args.eval_attack_eps = 8
    args.attack_steps = 10
    attack_cfg = setup_task.resolve_attack_cfg(args, eval=True)

    eps = 8 / 255
    assert attack_cfg.eps == eps

    expected_step_size = 1.5 * eps / args.attack_steps
    assert attack_cfg.step_size == expected_step_size


def test_resolve_attack_cfg_eval_name():
    args, _ = parse_args(additional_args=["foo"])
    args.attack = "targeted_pgd"
    attack_cfg = setup_task.resolve_attack_cfg(args, eval=True)

    assert attack_cfg.name == "pgd"


def test_adapt_model_patches_exception():
    model = xcit._create_xcit('xcit_small_12_p16_224')
    patch_size = 6
    assert isinstance(model, xcit.XCiT)
    with pytest.raises(AssertionError):
        adapt_model_patches(model, patch_size)


def test_adapt_model_patches_2():
    model = xcit._create_xcit('xcit_small_12_p16_224')
    patch_size = 2
    assert isinstance(model, xcit.XCiT)
    modified_model = adapt_model_patches(model, patch_size)
    assert modified_model.patch_embed.patch_size == patch_size
    assert modified_model.patch_embed.proj[0][0].stride == (1, 1)
    assert modified_model.patch_embed.proj[2][0].stride == (1, 1)
    assert modified_model.patch_embed.proj[4][0].stride == (1, 1)


def test_adapt_model_patches_4():
    model = xcit._create_xcit('xcit_small_12_p16_224')
    patch_size = 4
    assert isinstance(model, xcit.XCiT)
    modified_model = adapt_model_patches(model, patch_size)
    assert modified_model.patch_embed.patch_size == patch_size
    assert modified_model.patch_embed.proj[0][0].stride == (1, 1)
    assert modified_model.patch_embed.proj[2][0].stride == (1, 1)
    assert modified_model.patch_embed.proj[4][0].stride == (2, 2)


def test_adapt_model_patches_8():
    model = xcit._create_xcit('xcit_small_12_p16_224')
    patch_size = 8
    assert isinstance(model, xcit.XCiT)
    modified_model = adapt_model_patches(model, patch_size)
    assert modified_model.patch_embed.patch_size == patch_size
    assert modified_model.patch_embed.proj[0][0].stride == (1, 1)
    assert modified_model.patch_embed.proj[2][0].stride == (2, 2)
    assert modified_model.patch_embed.proj[4][0].stride == (2, 2)

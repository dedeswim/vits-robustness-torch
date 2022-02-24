import functools
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from torch import nn

AttackFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
TrainAttackFn = Callable[[nn.Module, torch.Tensor, torch.Tensor, int], torch.Tensor]
Boundaries = Tuple[float, float]
ProjectFn = Callable[[torch.Tensor, torch.Tensor, float, Boundaries], torch.Tensor]
InitFn = Callable[[torch.Tensor, float, ProjectFn, Boundaries], torch.Tensor]
EpsSchedule = Callable[[int], float]
ScheduleMaker = Callable[[float, int, int], EpsSchedule]
Norm = str


def project_linf(x: torch.Tensor, x_adv: torch.Tensor, eps: float, boundaries: Boundaries) -> torch.Tensor:
    clip_min, clip_max = boundaries
    d_x = torch.clamp(x_adv - x.detach(), -eps, eps)
    x_adv = torch.clamp(x + d_x, clip_min, clip_max)
    return x_adv


def init_linf(x: torch.Tensor, eps: float, project_fn: ProjectFn, boundaries: Boundaries) -> torch.Tensor:
    x_adv = x.detach() + torch.zeros_like(x.detach(), device=x.device).uniform_(-eps, eps) + 1e-5
    return project_fn(x, x_adv, eps, boundaries)


def pgd(model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        eps: float,
        step_size: float,
        steps: int,
        boundaries: Tuple[float, float],
        init_fn: InitFn,
        project_fn: ProjectFn,
        criterion: nn.Module,
        targeted: bool = False,
        num_classes: Optional[int] = None,
        random_targets: bool = False,
        logits_y: bool = False) -> torch.Tensor:
    local_project_fn = functools.partial(project_fn, eps=eps, boundaries=boundaries)
    x_adv = init_fn(x, eps, project_fn, boundaries)
    if random_targets:
        assert num_classes is not None
        y = torch.randint_like(y, 0, num_classes)
    if len(y.size()) > 1 and not logits_y:
        y = y.argmax(dim=-1)
    for _ in range(steps):
        x_adv.requires_grad_()
        loss = criterion(
            F.log_softmax(model(x_adv), dim=-1),
            y,
        )
        grad = torch.autograd.grad(loss, x_adv)[0]
        if targeted:
            # Minimize the loss if the attack is targeted
            x_adv = x_adv.detach() - step_size * torch.sign(grad)
        else:
            x_adv = x_adv.detach() + step_size * torch.sign(grad)

        x_adv = local_project_fn(x, x_adv)

    return x_adv


_ATTACKS = {"pgd": pgd, "targeted_pgd":
            functools.partial(pgd, targeted=True, random_targets=True)}
_INIT_PROJECT_FN: Dict[str, Tuple[InitFn, ProjectFn]] = {"linf": (init_linf, project_linf)}


def make_attack(attack: str,
                eps: float,
                step_size: float,
                steps: int,
                norm: Norm,
                boundaries: Tuple[float, float],
                criterion: nn.Module,
                device: Optional[torch.device] = None,
                **attack_kwargs) -> AttackFn:
    if attack != "autoattack":
        attack_fn = _ATTACKS[attack]
        init_fn, project_fn = _INIT_PROJECT_FN[norm]
        return functools.partial(attack_fn,
                                 eps=eps,
                                 step_size=step_size,
                                 steps=steps,
                                 boundaries=boundaries,
                                 init_fn=init_fn,
                                 project_fn=project_fn,
                                 criterion=criterion)

    def autoattack_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert isinstance(eps, float)
        adversary = AutoAttack(model, norm.capitalize(), eps=eps, device=device, **attack_kwargs)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        return x_adv  # type: ignore

    return autoattack_fn

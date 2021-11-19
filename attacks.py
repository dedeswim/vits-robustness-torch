import functools
from logging import log
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from torch import nn

AttackFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
Boundaries = Tuple[float, float]
ProjectFn = Callable[[torch.Tensor, torch.Tensor, float, Boundaries],
                     torch.Tensor]
InitFn = Callable[[torch.Tensor, float, ProjectFn, Boundaries], torch.Tensor]
Norm = str


def project_linf(x: torch.Tensor, x_adv: torch.Tensor, eps: float,
                 boundaries: Boundaries) -> torch.Tensor:
    clip_min, clip_max = boundaries
    d_x = torch.clamp(x_adv - x.detach(), -eps, eps)
    x_adv = torch.clamp(x + d_x, clip_min, clip_max)
    return x_adv


def init_linf(x: torch.Tensor, eps: float, project_fn: ProjectFn,
              boundaries: Boundaries) -> torch.Tensor:
    x_adv = x_adv = x.detach() + torch.zeros_like(
        x.detach(), device=x.device).uniform_(-eps, eps) + 1e-5
    return project_fn(x, x_adv, eps, boundaries)


def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
        step_size: float, steps: int, boundaries: Tuple[float, float],
        init_fn: InitFn, project_fn: ProjectFn,
        criterion: nn.Module) -> torch.Tensor:
    local_project_fn = functools.partial(project_fn,
                                         eps=eps,
                                         boundaries=boundaries)
    x_adv = init_fn(x, eps, project_fn, boundaries)
    for step in range(steps):
        x_adv.requires_grad_()
        loss = criterion(
            F.log_softmax(model(x_adv), dim=-1),
            y,
        )
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = local_project_fn(x, x_adv)

    return x_adv


_ATTACKS: Dict[str, AttackFn] = {"pgd": pgd}
_INIT_PROJECT_FN: Dict[str, Tuple[InitFn, ProjectFn]] = {
    "linf": (init_linf, project_linf)
}


def make_attack(attack: str,
                eps: float,
                step_size: float,
                steps: int,
                norm: Norm,
                boundaries: Tuple[float, float],
                criterion: nn.Module,
                device: Optional[torch.device] = None) -> AttackFn:
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

    def autoattack_fn(model: nn.Module, x: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        adversary = AutoAttack(model,
                               norm.capitalize(),
                               eps=eps,
                               device=device)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        return x_adv  # type: ignore

    return autoattack_fn


class AdvTrainingLoss(nn.Module):
    def __init__(self, attack: AttackFn, criterion: nn.Module):
        super().__init__()
        self.attack = attack
        self.criterion = criterion

    def forward(
            self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_adv = self.attack(model, x, y.argmax(dim=-1))
        logits, logits_adv = model(x), model(x_adv)
        loss = self.criterion(logits_adv, y)
        return loss, logits, logits_adv


class TRADESLoss(nn.Module):
    def __init__(self, attack: AttackFn, natural_criterion: nn.Module,
                 beta: float):
        super().__init__()
        self.attack = attack
        self.natural_criterion = natural_criterion
        self.kl_criterion = nn.KLDivLoss(reduction="sum")
        self.beta = beta

    def forward(
            self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        # model.eval()  # FIXME: understand why with eval the gradient of BatchNorm crashes
        output_softmax = F.softmax(model(x.detach()), dim=-1)
        x_adv = self.attack(model, x, output_softmax)
        model.train()
        logits, logits_adv = model(x), model(x_adv)
        loss_natural = self.natural_criterion(logits, y)
        loss_robust = (1.0 / batch_size) * self.kl_criterion(
            F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
        loss = loss_natural + self.beta * loss_robust
        return loss, logits, logits_adv

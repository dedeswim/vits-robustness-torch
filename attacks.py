import functools
from typing import Callable, Dict, Tuple

import timm.optim
import torch
import torch.nn.functional as F
from torch import nn, optim

AttackFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]
Boundaries = Tuple[float, float]
ProjectFn = Callable[[torch.Tensor, torch.Tensor, float, Boundaries], torch.Tensor]
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
    x_adv = x.detach() + 0.001 * torch.rand_like(x.detach(), device=x.device) + 1e-5  # TODO: add eps * rand?
    return project_fn(x, x_adv, eps, boundaries)


def pgd(model: nn.Module, x: torch.Tensor, eps: float, step_size: float, steps: int,
        boundaries: Tuple[float, float], init_fn: InitFn, project_fn: ProjectFn) -> torch.Tensor:
    criterion_kl = nn.KLDivLoss(reduction="sum")
    local_project_fn = functools.partial(project_fn, eps=eps, boundaries=boundaries)
    x_adv = nn.Parameter(init_fn(x, eps, project_fn, boundaries))
    output_softmax = F.softmax(model(x.detach()), dim=-1)
    for step in range(steps):
        # print(f"{step=}")
        x_adv.requires_grad_()
        loss_kl = criterion_kl(
            F.log_softmax(model(x_adv), dim=-1),
            output_softmax,
        )
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = local_project_fn(x, x_adv)

    return x_adv


_ATTACKS = {"pgd": pgd}
_INIT_PROJECT_FN: Dict[str, Tuple[InitFn, ProjectFn]] = {"linf": (init_linf, project_linf)}


def make_attack(attack: str, eps: float, step_size: float, steps: int,
                norm: Norm, boundaries: Tuple[float, float]) -> AttackFn:
    attack_fn = _ATTACKS[attack]
    init_fn, project_fn = _INIT_PROJECT_FN[norm]
    return functools.partial(attack_fn,
                             eps=eps,
                             step_size=step_size,
                             steps=steps,
                             boundaries=boundaries,
                             init_fn=init_fn,
                             project_fn=project_fn)


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
        x_adv = self.attack(model, x)
        model.train()
        logits, logits_adv = model(x), model(x_adv)
        loss_natural = self.natural_criterion(logits, y)
        loss_robust = (1.0 / batch_size) * self.kl_criterion(
            F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
        loss = loss_natural + self.beta * loss_robust
        # print(f"TRADES loss: {loss}")
        return loss, logits, logits_adv

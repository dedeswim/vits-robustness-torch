import functools
from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

AttackFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]
Norm = Union[int, float]


def pgd(model, x, eps, step_size, steps, norm, boundaries):
    clip_min, clip_max = boundaries
    criterion_kl = nn.KLDivLoss(size_average=False)
    x_adv = (x.detach() + 0.001 * torch.randn_like(x.detach()))
    if norm == "linf":
        for _ in range(steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x), dim=1),
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        raise ValueError("Norm not supported")

    return x_adv


_ATTACKS = {"pgd": pgd}


def make_attack(attack: str, eps: float, step_size: float, steps: int,
                norm: Norm, boundaries: Tuple[float, float]) -> AttackFn:
    attack_fn = _ATTACKS[attack]
    return functools.partial(attack_fn,
                             eps=eps,
                             step_size=step_size,
                             nb_iter=steps,
                             norm=norm,
                             boundaries=boundaries)


class TRADESLoss(nn.Module):
    def __init__(self, attack: AttackFn, natural_criterion: nn.Module,
                 beta: float):
        super().__init__()
        self.attack = attack
        self.natural_criterion = natural_criterion
        self.kl_criterion = nn.KLDivLoss(size_average=False)
        self.beta = beta

    def forward(
            self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        model.eval()
        x_adv = self.attack(model, x)
        model.train()
        logits, logits_adv = model(x), model(x_adv)
        loss_natural = self.natural_criterion(logits, y)
        loss_robust = (1.0 / batch_size) * self.kl_criterion(
            F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1))
        loss = loss_natural + self.beta * loss_robust
        return loss, logits, logits_adv

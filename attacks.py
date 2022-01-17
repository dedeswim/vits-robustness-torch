import functools
import math
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
        random_targets: bool = False) -> torch.Tensor:
    local_project_fn = functools.partial(project_fn, eps=eps, boundaries=boundaries)
    x_adv = init_fn(x, eps, project_fn, boundaries)
    if random_targets:
        assert num_classes is not None
        y = torch.randint_like(y, 0, num_classes)
    if len(y.size()) > 1:
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


def make_sine_schedule(final: float, warmup: int, zero_eps_epochs: int) -> Callable[[int], float]:
    def sine_schedule(step: int) -> float:
        if step < zero_eps_epochs:
            return 0.0
        if step < warmup:
            return 0.5 * final * (1 + math.sin(math.pi * ((step - zero_eps_epochs) / warmup - 0.5)))
        return final

    return sine_schedule


def make_linear_schedule(final: float, warmup: int, zero_eps_epochs: int) -> Callable[[int], float]:
    def linear_schedule(step: int) -> float:
        if step < zero_eps_epochs:
            return 0.0
        if step < warmup:
            return (step - zero_eps_epochs) / warmup * final
        return final

    return linear_schedule


_SCHEDULES: Dict[str, ScheduleMaker] = {
    "linear": make_linear_schedule,
    "sine": make_sine_schedule,
    "constant": (lambda eps, _1, _2: (lambda _: eps))
}


def make_train_attack(attack_name: str, schedule: str, final_eps: float, period: int, zero_eps_epochs: int,
                      step_size: float, steps: int, norm: Norm, boundaries: Tuple[float, float],
                      criterion: nn.Module, num_classes: int) -> TrainAttackFn:
    if attack_name in {"ll", "soft-labels"}:
        attack_mode: Optional[str] = attack_name
        attack_name = "pgd"
    else:
        attack_mode = None
    attack_fn = _ATTACKS[attack_name]
    init_fn, project_fn = _INIT_PROJECT_FN[norm]
    schedule_fn = _SCHEDULES[schedule](final_eps, period, zero_eps_epochs)

    def attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor:
        eps = schedule_fn(step)
        if attack_mode == "ll":
            with torch.no_grad():
                y = model(x).argmin(dim=-1)
        elif attack_mode == "soft-labels":
            with torch.no_grad():
                y = model(x)
        return attack_fn(model,
                         x,
                         y,
                         eps,
                         step_size=step_size,
                         steps=steps,
                         boundaries=boundaries,
                         init_fn=init_fn,
                         project_fn=project_fn,
                         criterion=criterion,
                         num_classes=num_classes)

    return attack


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

    def autoattack_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert isinstance(eps, float)
        adversary = AutoAttack(model, norm.capitalize(), eps=eps, device=device, verbose=False)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        return x_adv  # type: ignore

    return autoattack_fn


class AdvTrainingLoss(nn.Module):
    def __init__(self, attack: TrainAttackFn, criterion: nn.Module):
        super().__init__()
        self.attack = attack
        self.criterion = criterion

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_adv = self.attack(model, x, y, epoch)
        logits, logits_adv = model(x), model(x_adv)
        loss = self.criterion(logits_adv, y)
        return loss, logits, logits_adv


class TRADESLoss(nn.Module):
    def __init__(self, attack: TrainAttackFn, natural_criterion: nn.Module, beta: float):
        super().__init__()
        self.attack = attack
        self.natural_criterion = natural_criterion
        self.kl_criterion = nn.KLDivLoss(reduction="sum")
        self.beta = beta

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        # model.eval()  # FIXME: understand why with eval the gradient
        # of BatchNorm crashes
        output_softmax = F.softmax(model(x.detach()), dim=-1)
        x_adv = self.attack(model, x, output_softmax, epoch)
        model.train()
        logits, logits_adv = model(x), model(x_adv)
        loss_natural = self.natural_criterion(logits, y)
        loss_robust = (1.0 / batch_size) * self.kl_criterion(F.log_softmax(logits_adv, dim=1),
                                                             F.softmax(logits, dim=1))
        loss = loss_natural + self.beta * loss_robust
        return loss, logits, logits_adv

import functools
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from timm.bits import DeviceEnv
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
    x_adv = x.detach() + torch.empty_like(x.detach(), device=x.device).uniform_(-eps, eps) + 1e-5
    return project_fn(x, x_adv, eps, boundaries)


def init_l2(x: torch.Tensor, eps: float, project_fn: ProjectFn, boundaries: Boundaries) -> torch.Tensor:
    x_adv = x.detach() + torch.empty_like(x.detach(), device=x.device).normal_(-eps, eps) + 1e-5
    return project_fn(x, x_adv, eps, boundaries)


def project_l2(x: torch.Tensor, x_adv: torch.Tensor, eps: float, boundaries: Boundaries) -> torch.Tensor:
    clip_min, clip_max = boundaries
    d_x = x_adv - x.detach()
    d_x_norm = d_x.renorm(p=2, dim=0, maxnorm=eps)
    x_adv = torch.clamp(x + d_x_norm, clip_min, clip_max)
    return x_adv


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
        logits_y: bool = False,
        take_sign=True,
        dev_env: Optional[DeviceEnv] = None) -> torch.Tensor:
    local_project_fn = functools.partial(project_fn, eps=eps, boundaries=boundaries)
    x_adv = init_fn(x, eps, project_fn, boundaries)
    if random_targets:
        assert num_classes is not None
        y = torch.randint_like(y, 0, num_classes, device=y.device)
    if len(y.size()) > 1 and not logits_y:
        y = y.argmax(dim=-1)
    for _ in range(steps):
        x_adv.requires_grad_()
        loss = criterion(
            F.log_softmax(model(x_adv), dim=-1),
            y,
        )
        grad = torch.autograd.grad(loss, x_adv)[0]
        if take_sign:
            d_x = torch.sign(grad)
        else:
            d_x = grad
        if targeted:
            # Minimize the loss if the attack is targeted
            x_adv = x_adv.detach() - step_size * d_x
        else:
            x_adv = x_adv.detach() + step_size * d_x

        x_adv = local_project_fn(x, x_adv)

        if dev_env is not None:
            # Mark step here to keep XLA program size small and speed-up compilation time
            # It also seems to improve overall speed when `steps` > 1.
            dev_env.mark_step()

    return x_adv


_ATTACKS = {"pgd": pgd, "targeted_pgd": functools.partial(pgd, targeted=True, random_targets=True)}
_INIT_PROJECT_FN: Dict[str, Tuple[InitFn, ProjectFn]] = {
    "linf": (init_linf, project_linf),
    "l2": (init_l2, project_l2)
}


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
                      criterion: nn.Module, num_classes: int, logits_y: bool, **kwargs) -> TrainAttackFn:
    attack_fn = _ATTACKS[attack_name]
    init_fn, project_fn = _INIT_PROJECT_FN[norm]
    schedule_fn = _SCHEDULES[schedule](final_eps, period, zero_eps_epochs)

    def attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor:
        eps = schedule_fn(step)
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
                         num_classes=num_classes,
                         logits_y=logits_y,
                         **kwargs)

    return attack


def make_attack(attack: str,
                eps: float,
                step_size: float,
                steps: int,
                norm: Norm,
                boundaries: Tuple[float, float],
                criterion: nn.Module,
                device: Optional[torch.device] = None,
                **attack_kwargs) -> AttackFn:
    if attack not in {"autoattack", "apgd-ce"}:
        attack_fn = _ATTACKS[attack]
        init_fn, project_fn = _INIT_PROJECT_FN[norm]
        return functools.partial(attack_fn,
                                 eps=eps,
                                 step_size=step_size,
                                 steps=steps,
                                 boundaries=boundaries,
                                 init_fn=init_fn,
                                 project_fn=project_fn,
                                 criterion=criterion,
                                 **attack_kwargs)
    if attack in {"apgd-ce"}:
        attack_kwargs["version"] = "custom"
        attack_kwargs["attacks_to_run"] = [attack]
        if "dev_env" in attack_kwargs:
            del attack_kwargs["dev_env"]

    def autoattack_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert isinstance(eps, float)
        adversary = AutoAttack(model, norm.capitalize(), eps=eps, device=device, **attack_kwargs)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        return x_adv  # type: ignore

    return autoattack_fn


@dataclass
class AttackCfg:
    name: str
    eps: float
    eps_schedule: str
    eps_schedule_period: int
    zero_eps_epochs: int
    step_size: float
    steps: int
    norm: str
    boundaries: Tuple[float, float]


class AdvTrainingLoss(nn.Module):

    def __init__(self,
                 attack_cfg: AttackCfg,
                 natural_criterion: nn.Module,
                 dev_env: DeviceEnv,
                 num_classes: int,
                 eval_mode: bool = False):
        super().__init__()
        self.criterion = natural_criterion
        self.attack = make_train_attack(attack_cfg.name,
                                        attack_cfg.eps_schedule,
                                        attack_cfg.eps,
                                        attack_cfg.eps_schedule_period,
                                        attack_cfg.zero_eps_epochs,
                                        attack_cfg.step_size,
                                        attack_cfg.steps,
                                        attack_cfg.norm,
                                        attack_cfg.boundaries,
                                        criterion=nn.NLLLoss(reduction="sum"),
                                        num_classes=num_classes,
                                        logits_y=False,
                                        dev_env=dev_env)
        self.eval_mode = eval_mode

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.eval_mode:
            model.eval()
        x_adv = self.attack(model, x, y, epoch)
        model.train()
        logits, logits_adv = model(x), model(x_adv)
        loss = self.criterion(logits_adv, y)
        return loss, logits, logits_adv


class TRADESLoss(nn.Module):
    """Adapted from https://github.com/yaodongyu/TRADES/blob/master/trades.py#L17"""
    def __init__(self,
                 attack_cfg: AttackCfg,
                 natural_criterion: nn.Module,
                 beta: float,
                 dev_env: DeviceEnv,
                 num_classes: int,
                 eval_mode: bool = False):
        super().__init__()
        self.attack = make_train_attack(attack_cfg.name,
                                        attack_cfg.eps_schedule,
                                        attack_cfg.eps,
                                        attack_cfg.eps_schedule_period,
                                        attack_cfg.zero_eps_epochs,
                                        attack_cfg.step_size,
                                        attack_cfg.steps,
                                        attack_cfg.norm,
                                        attack_cfg.boundaries,
                                        criterion=nn.KLDivLoss(reduction="sum"),
                                        num_classes=num_classes,
                                        logits_y=True,
                                        dev_env=dev_env)
        self.natural_criterion = natural_criterion
        self.kl_criterion = nn.KLDivLoss(reduction="sum")
        self.beta = beta
        self.eval_mode = eval_mode

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        # Avoid setting the model in eval mode if on XLA (it crashes)
        if self.eval_mode:
            model.eval()  # FIXME: understand why with eval the gradient
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

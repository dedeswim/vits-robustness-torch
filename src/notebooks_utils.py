import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms.functional as F
from robustbench.loaders import CustomImageFolder
from torch import nn
from torch.utils import data
from torchvision import transforms

from src import attacks


def get_data(n_examples, data_dir="~/imagenet/", seed=0):
    torch.manual_seed(seed)
    preprocessing = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224),
         transforms.ToTensor()])
    imagenet = CustomImageFolder(data_dir + '/val', preprocessing)
    test_loader = data.DataLoader(imagenet, batch_size=n_examples, shuffle=True, num_workers=4)
    x, y, _ = next(iter(test_loader))
    return x, y


def get_adv_examples(model, device, n_examples, eps=4 / 255, pgd_steps=100, seed=0):
    x, y = get_data(n_examples, seed=seed)
    attack_criterion = nn.NLLLoss(reduction="sum")
    attack = attacks.make_attack("pgd", eps, 1.5 * eps / pgd_steps, pgd_steps, "linf", (0, 1),
                                 attack_criterion)
    x, y = x.to(device), y.to(device)
    model = model.to(device)
    adv_x = attack(model, x, y)
    return x, adv_x, y


def get_synth_examples(model,
                       device,
                       n_examples,
                       eps=15,
                       pgd_steps=100,
                       input_size=(3, 224, 224),
                       num_classes=1000,
                       random_start=True,
                       reinforce=False,
                       seed=0):
    # Fix seed to start from the same random samples
    if random_start:
        torch.manual_seed(seed)
        x = torch.empty((n_examples, *input_size), device=device).uniform_()
        y = torch.randint(low=0, high=num_classes, size=(n_examples, ), device=device)
    else:
        x, y = get_data(n_examples)
        x, y = x.to(device), y.to(device)
    attack_criterion = nn.NLLLoss(reduction="sum")
    attack = attacks.make_attack("pgd",
                                 eps,
                                 1.5 * eps / pgd_steps,
                                 pgd_steps,
                                 "linf", (0, 1),
                                 attack_criterion,
                                 targeted=True,
                                 random_targets=not random_start and not reinforce,
                                 num_classes=num_classes)
    adv_x = attack(model, x, y)
    return x, adv_x, y


def show_grid(xs, ncols=4, cmap=None, labels=None, filename=None, axes_pad=1.5):
    if len(xs.shape) > 3:
        xs = [np.asarray(F.to_pil_image(x)) for x in xs]
    fig = plt.figure(figsize=(30, 30))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(len(xs) // ncols, ncols),  # creates 2x2 grid of axes
        axes_pad=axes_pad,  # pad between axes in inch.
    )

    for i, (ax, im) in enumerate(zip(grid, xs)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap)
        if labels is not None:
            ax.set_title(labels[i], fontdict={'fontsize': 20}, pad=20)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if filename is not None:
        fig.savefig(filename)

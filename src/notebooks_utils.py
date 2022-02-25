import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms.functional as F
from robustbench.loaders import CustomImageFolder
from torch import nn
from torch.utils import data
from torchvision import transforms

import src.attacks as attacks


def get_data(n_examples, data_dir="~/imagenet/", crop_pct=1.0):
    torch.manual_seed(0)
    res = 224
    resize_res = int(res / crop_pct)
    preprocessing = transforms.Compose(
        [transforms.Resize(resize_res), transforms.CenterCrop(res),
         transforms.ToTensor()])
    imagenet = CustomImageFolder(data_dir + '/val', preprocessing)
    test_loader = data.DataLoader(imagenet, batch_size=n_examples, shuffle=True, num_workers=4)
    x, y, _ = next(iter(test_loader))
    return x, y


def get_adv_examples(model, device, n_examples, eps=4 / 255, pgd_steps=100, crop_pct=1.0):
    x, y = get_data(n_examples, crop_pct=crop_pct)
    attack_criterion = nn.NLLLoss(reduction="sum")
    attack = attacks.make_attack("pgd", eps, 1.5 * eps / pgd_steps, pgd_steps, "linf", (0, 1),
                                 attack_criterion)
    x, y = x.to(device), y.to(device)
    model = model.to(device)
    adv_x = attack(model, x, y)
    return x, adv_x, y


def show_grid(xs, ncols=4, cmap=None):
    if len(xs.shape) > 3:
        xs = [np.asarray(F.to_pil_image(x)) for x in xs]
    fig = plt.figure(figsize=(30, 30))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(len(xs) // ncols, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    for ax, im in zip(grid, xs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

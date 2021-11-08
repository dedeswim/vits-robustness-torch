from typing import Optional
from torchvision import datasets

from timm.data.dataset_factory import create_dataset


_DATASETS = {
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100
}


def my_create_dataset(name: str,
                      root: str,
                      split: str = 'validation',
                      search_split: bool = True,
                      is_training: bool = False,
                      batch_size: Optional[int] = None,
                      **kwargs):
    """Wraps timm's create dataset to use also PyTorch's datasets"""
    name = name.lower()
    if not name.startswith('torch'):
        return create_dataset(name, root, split, search_split, is_training,
                              batch_size, **kwargs)

    dataset_name = name.split('/')[-1]
    return _DATASETS[dataset_name](root, train=is_training, download=True)

import os
import tempfile

from pathlib import Path

import tensorflow as tf
import torch

def main(output_dir, model_name, checkpoint_path, dataset, eps=None):
    print(model_name, dataset, eps)
    checkpoint_path = f"gs://{checkpoint_path}/best.pth.tar"
    with tempfile.TemporaryDirectory() as dst:
        local_checkpoint_path = os.path.join(dst, os.path.basename(checkpoint_path))
        tf.io.gfile.copy(checkpoint_path, local_checkpoint_path)
        state_dict = torch.load(local_checkpoint_path)
    state_dict_to_keep = state_dict["model"]
    if eps is not None:
        filename = f"{model_name}-{dataset}-eps-{eps}.pth.tar"
    else:
        filename = f"{model_name}-{dataset}.pth.tar"
    output_path = Path(output_dir) / dataset / filename

    torch.save(state_dict_to_keep, output_path)


if __name__ == "__main__":
    models_to_save = [
        # ImageNet eps 4
        # ("robust-vits/xcit", "xcit-s12", "ImageNet", 4),
        # ("robust-vits/xcit-big-5", "xcit-m12", "ImageNet", 4),
        # ("robust-vits-us/xcit-big-50", "xcit-l12", "ImageNet", 4),
        # ("robust-vits-us/model-smoothness-4", "resnet50_gelu", "ImageNet", 4),
        # ("robust-vits-us/convnext-adv-training-1", "convnext_tiny", "ImageNet", 4),
        # # ImageNet eps 8
        # ("robust-vits/xcit-adv-pretraining-3","xcit-s12", "ImageNet", 8),
        # ("robust-vits-us/xcit-adv-pretraining-1", "xcit-m12", "ImageNet", 8),
        # ("robust-vits-us/xcit-adv-pretraining-5", "xcit-l12", "ImageNet", 8),
        # ("robust-vits-us/xcit-adv-pretraining-9", "resnet50_gelu", "ImageNet", 8),
        # ("robust-vits-us/xcit-adv-pretraining-10", "convnext_tiny", "ImageNet", 8),
        # CIFAR-10
        ("robust-vits/xcit-adv-finetuning-tpu_cifar10_ablations_only_randErasing-1", "xcit-s12", "CIFAR-10", None),
        ("robust-vits/xcit-adv-finetuning-tpu_cifar10_large-6", "xcit-m12", "CIFAR-10", None),
        ("robust-vits-us/xcit-adv-finetuning-tpu_cifar10_large-3", "xcit-l12", "CIFAR-10", None),
        ("robust-vits-us/xcit-adv-finetuning-tpu_cifar10_large-22", "resnet_50", "CIFAR-10", None),
        # CIFAR-100
        ("robust-vits-us/xcit-adv-finetuning-tpu_cifar100_large-5", "xcit-s12", "CIFAR-100", None),
        ("robust-vits/xcit-adv-finetuning-tpu_cifar100_large-2", "xcit-m12", "CIFAR-100", None),
        ("robust-vits-us/xcit-adv-finetuning-tpu_cifar100_large-2", "xcit-l12", "CIFAR-100", None),
        ("robust-vits-us/xcit-adv-finetuning-tpu_cifar100_large-15", "resnet_50", "CIFAR-100", None),
        # Oxford Flowers
        ("robust-vits-us/xcit-adv-finetuning-hi-res-9", "xcit-s12", "Oxford Flowers", None),
        ("robust-vits-us/xcit-adv-finetuning-hi-res-13", "resnet_50", "Oxford Flowers", None),
        # Caltech101
        ("robust-vits-us/xcit-adv-finetuning-hi-res-10", "xcit-s12", "Caltech101", None),
        ("robust-vits-us/xcit-adv-finetuning-hi-res-11", "resnet_50", "Caltech101", None)
    ]

    for checkpoint_path, model_name, dataset, eps in models_to_save:
        main("drive_checkpoints", model_name, checkpoint_path, dataset, eps)
    
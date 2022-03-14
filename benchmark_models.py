from typing import Tuple

import thop
import torch
import timm

from src import models as _  # Necessary to register models in timm


def benchmark_model(model_name: str, input_size: Tuple[int, ...], num_classes, verbose=True):
    model = timm.create_model(model_name, num_classes=num_classes)
    x = torch.randn(1, *input_size)
    macs, params = thop.profile(model, inputs=(x, ), verbose=False)
    if verbose:
        macs_p, params_p = thop.clever_format([macs, params], "%.3f")
        print(f"Model: {model_name}, {input_size = }, {num_classes = } - {macs_p} MACs, {params_p} params")
    return macs, params


if __name__ == "__main__":
    input_size = (3, 224, 224)
    models = ["resnet18", "resnet50", "xcit_small_12_p16_224", "wide_resnet50_2"]
    num_classes = 1000
    for model in models:
        benchmark_model(model, input_size, num_classes)

    print()

    input_size = (3, 32, 32)
    all_num_classes = [10, 100]
    models = [
        "xcit_small_12_p8_32", "preact_resnet_18", "wide_resnet28_10", "wide_resnet34_10", "wide_resnet34_20",
        "wide_resnet70_16", "wide_resnet106_16"
    ]
    for num_classes in all_num_classes:
        for model in models:
            benchmark_model(model, input_size, num_classes)
        print()

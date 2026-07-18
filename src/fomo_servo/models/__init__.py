"""Public MobileNetV2-lite FOMO model APIs."""

from .fomo import (
    OUTPUT_STRIDE,
    FOMONet,
    ModelConfigurationError,
    build_fomo_model,
    count_trainable_parameters,
)
from .metadata import describe_model
from .pretrained import (
    PretrainedLoadReport,
    PretrainedWeightsError,
    load_ei_mobilenet_v2_backbone,
)
from .mobilenet_v2_fomo import (
    BLOCK_6_EXPAND_RELU,
    STANDARD_MOBILENET_V2_BLOCK_SPECS,
    MobileNetV2BlockSpec,
    MobileNetV2FOMOBackbone,
    MobileNetV2FOMONet,
)
from .alternative_fomo import (
    MOBILENET_V3_SMALL_CUT_POINT,
    SQUEEZENET1_1_CUT_POINT,
    MobileNetV3SmallFOMONet,
    SqueezeNet1_1FOMONet,
    TorchvisionFeatureBackbone,
)

__all__ = [
    "OUTPUT_STRIDE",
    "FOMONet",
    "ModelConfigurationError",
    "build_fomo_model",
    "count_trainable_parameters",
    "describe_model",
    "PretrainedLoadReport",
    "PretrainedWeightsError",
    "load_ei_mobilenet_v2_backbone",
    "BLOCK_6_EXPAND_RELU",
    "STANDARD_MOBILENET_V2_BLOCK_SPECS",
    "MobileNetV2BlockSpec",
    "MobileNetV2FOMOBackbone",
    "MobileNetV2FOMONet",
    "MOBILENET_V3_SMALL_CUT_POINT",
    "SQUEEZENET1_1_CUT_POINT",
    "MobileNetV3SmallFOMONet",
    "SqueezeNet1_1FOMONet",
    "TorchvisionFeatureBackbone",
]

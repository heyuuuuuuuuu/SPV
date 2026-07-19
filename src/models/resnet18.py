"""
ResNet18SPV: 基于 ResNet-18 的 SPV 识别模型

方式 (默认): 将 1 通道灰度图复制为 3 通道，使用标准 ResNet18
方式 2 (in_channels=1): 修改第一层卷积接收 1 通道输入
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18SPV(nn.Module):
    """
    ResNet18 适配 SPV 汉字识别

    Args:
        num_classes:  输出类别数
        in_channels:  输入通道数 (1 或 3)
        pretrained:   是否加载 ImageNet 预训练权重 (仅当 in_channels=3 时有效)
    """

    def __init__(
        self,
        num_classes: int = 999,
        in_channels: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.use_replicate = (in_channels == 3)

        if pretrained and in_channels == 3:
            self.backbone = resnet18(weights="IMAGENET1K_V1")
        elif pretrained and in_channels == 1:
            self.backbone = resnet18(weights=None)
            self._replace_first_conv(in_channels)
        else:
            self.backbone = resnet18(weights=None)
            if in_channels == 1:
                self._replace_first_conv(in_channels)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def _replace_first_conv(self, in_channels: int):
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        nn.init.kaiming_normal_(
            self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_replicate and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)

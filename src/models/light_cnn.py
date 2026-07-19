"""
LightCNN: 轻量级卷积神经网络 (SPV 汉字识别)

架构:
  Conv2d(1→64, 3×3) → BN → ReLU → MaxPool(4)
  Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2)
  Conv2d(128→256, 3×3) → BN → ReLU → MaxPool(2)
  AdaptiveAvgPool(1) → Dropout(0.3) → FC(256 → num_classes)

参数量约 0.5M (num_classes=999)
"""

import torch
import torch.nn as nn


class LightCNN(nn.Module):
    """轻量级 CNN，适用于 SPV 汉字识别"""

    def __init__(
        self,
        num_classes: int = 999,
        in_channels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1×128×128 → 64×64×64
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),

            # Block 2: 64×64×64 → 128×32×32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 128×32×32 → 256×16×16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

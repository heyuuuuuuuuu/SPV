"""
模型定义: LightCNN 和 ResNet18 (SPV 汉字识别)

用法:
  from models import LightCNN, ResNet18SPV

  model = LightCNN(num_classes=999)
  model = ResNet18SPV(num_classes=999, in_channels=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ═══════════════════════════════════════════════════════════
# LightCNN: 轻量级卷积神经网络
# ═══════════════════════════════════════════════════════════

class LightCNN(nn.Module):
    """
    轻量级 CNN, 适用于 SPV 汉字识别

    架构:
      Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(2)
      Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2)
      Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2)
      Conv2d(128→256, 3×3) → BN → ReLU → AdaptiveAvgPool(1)
      Dropout(0.3) → FC(256 → num_classes)

    参数量约 0.5M (num_classes=999)
    """

    def __init__(
        self,
        num_classes: int = 999,
        in_channels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1×128×128 → 32×64×64
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),  # 64×64

            # # Block 1: 1×128×128 → 32×64×64
            # nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),  # 64×64
            # # Block 2: 32×64×64 → 64×32×32
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),  # 32×32

            # Block 3: 64×32×32 → 128×16×16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16×16

            # Block 4: 128×16×16 → 256×8×8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8×8
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # → 256×1×1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

        # 初始化权重
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
        """
        Args:
            x: (B, C, H, W)

        Returns:
            logits: (B, num_classes)
        """
        #print("zhelushikaishide:",x.shape)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        #print("zuizhongshuchuded:",x.shape)
        return x


# ═══════════════════════════════════════════════════════════
# ResNet18SPV: 基于 ResNet-18 的 SPV 识别模型
# ═══════════════════════════════════════════════════════════

class ResNet18SPV(nn.Module):
    """
    ResNet18 适配 SPV 汉字识别

    方式 (默认): 将 1 通道灰度图复制为 3 通道, 使用标准 ResNet18
    方式 2 (in_channels=1): 修改第一层卷积接收 1 通道输入

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

        # 加载 ResNet18
        if pretrained and in_channels == 3:
            self.backbone = resnet18(weights="IMAGENET1K_V1")
        elif pretrained and in_channels == 1:
            # 1 通道时无法直接加载预训练权重, 手动构建
            self.backbone = resnet18(weights=None)
            self._replace_first_conv(in_channels)
        else:
            self.backbone = resnet18(weights=None)
            if in_channels == 1:
                self._replace_first_conv(in_channels)

        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def _replace_first_conv(self, in_channels: int):
        """将 ResNet18 的第一层卷积替换为接收 in_channels 输入"""
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # Kaiming 初始化
        nn.init.kaiming_normal_(
            self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) 或 (B, 3, H, W)

        Returns:
            logits: (B, num_classes)
        """
        # 如果输入是 1 通道且模型期望 3 通道, 复制灰度图
        if self.use_replicate and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        return self.backbone(x)


# ═══════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════

def create_model(
    model_name: str,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """
    根据名称创建模型

    Args:
        model_name:  "light_cnn" 或 "resnet18"
        num_classes: 分类数
        **kwargs:    传递给模型构造函数的额外参数

    Returns:
        nn.Module
    """
    model_name = model_name.lower()

    if model_name == "light_cnn":
        return LightCNN(
            num_classes=num_classes,
            in_channels=kwargs.get("in_channels", 1),
            dropout=kwargs.get("dropout", 0.3),
        )

    elif model_name == "resnet18":
        return ResNet18SPV(
            num_classes=num_classes,
            in_channels=kwargs.get("in_channels", 3),
            pretrained=kwargs.get("pretrained", True),
        )

    else:
        raise ValueError(f"未知模型: {model_name}, 可选: light_cnn, resnet18")


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 快速测试模型输出维度
    print("测试模型输出维度...")

    for name in ["light_cnn", "resnet18"]:
        model = create_model(name, num_classes=999)
        model.eval()

        for in_ch in [1, 3]:
            try:
                if name == "light_cnn":
                    model_test = create_model(name, num_classes=999, in_channels=in_ch)
                else:
                    model_test = create_model(name, num_classes=999, in_channels=in_ch, pretrained=False)

                x = torch.randn(2, in_ch, 128, 128)
                with torch.no_grad():
                    y = model_test(x)
                print(f"  {name} (in={in_ch}): input {tuple(x.shape)} -> output {tuple(y.shape)}  [OK]")

                # 参数量
                n_params = sum(p.numel() for p in model_test.parameters())
                print(f"    params: {n_params / 1e6:.2f}M")

            except Exception as e:
                print(f"  {name} (in={in_ch}): ERROR - {e}")

    print("完成!")

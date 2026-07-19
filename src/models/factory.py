"""
模型工厂函数: 根据名称创建模型

用法:
  from src.models import create_model
  model = create_model("light_cnn", num_classes=999)
  model = create_model("resnet18", num_classes=999, in_channels=3)
"""

import torch
import torch.nn as nn
from .light_cnn import LightCNN
from .resnet18 import ResNet18SPV


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


if __name__ == "__main__":
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

                n_params = sum(p.numel() for p in model_test.parameters())
                print(f"    params: {n_params / 1e6:.2f}M")

            except Exception as e:
                print(f"  {name} (in={in_ch}): ERROR - {e}")

    print("完成!")

"""Arquitetura da CNN para classificação binária (maligno vs. benigno)."""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class MammographyCNN(nn.Module):
    """CNN simples de 4 blocos convolucionais + cabeça densa.

    Entrada: (batch, 1, 224, 224) em tons de cinza.
    Saída: 1 logit (BCEWithLogitsLoss) — probabilidade de ser maligno.
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


def build_resnet18_transfer(unfreeze_from: str = "layer3", dropout: float = 0.4) -> nn.Module:
    """ResNet18 pré-treinada no ImageNet, com a cabeça trocada por um classificador
    binário e apenas as camadas finais (`unfreeze_from` em diante) treináveis.

    Fine-tuning parcial é a prática padrão para datasets médicos pequenos: mantém os
    filtros iniciais (bordas, texturas) já bem aprendidos no ImageNet, e adapta apenas
    as camadas mais específicas às características de mamografia.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    unfreeze = False
    for name, module in model.named_children():
        if name == unfreeze_from:
            unfreeze = True
        for param in module.parameters():
            param.requires_grad = unfreeze

    in_features = model.fc.in_features
    # model.fc é substituído por um módulo novo, cujos parâmetros já nascem com
    # requires_grad=True por padrão do PyTorch — treinável independentemente do
    # ponto de corte (`unfreeze_from`) usado acima.
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1),
    )

    return model

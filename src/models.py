"""Trzy porównywane architektury: CNN od zera, EfficientNetB0 i ResNet50.

Każdy model przyjmuje obraz w skali szarości (B x 1 x H x W, wartości 0-1),
a normalizację robi sam - dzięki temu GUI i Grad-CAM podają ten sam tensor.
"""
import torch
import torch.nn as nn
from torchvision import models

N_CLASSES = 4
# średnia i odchylenie jasności pikseli zbioru treningowego (notatnik 01_eda)
MRI_MEAN, MRI_STD = 0.18, 0.20
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Punkt odniesienia: 4 bloki Conv-BN-ReLU-Pool, uczony od zera na 1 kanale."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                      conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(256, N_CLASSES))

    def forward(self, x):
        x = (x - MRI_MEAN) / MRI_STD
        return self.head(self.features(x))

    def cam_layer(self):
        return self.features[-1]


class Pretrained(nn.Module):
    """Sieć wstępnie wytrenowana na ImageNet z nową głowicą na 4 klasy.

    Obraz szary kopiujemy na 3 kanały i normalizujemy statystykami ImageNet,
    bo takiego wejścia oczekują wagi."""

    def __init__(self, name, dropout=0.3):
        super().__init__()
        self.name = name
        if name == 'efficientnet_b0':
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            n_feat = net.classifier[1].in_features
            net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_feat, N_CLASSES))
            # bloki od wejścia do wyjścia: stem, 7 etapów MBConv, końcowa konwolucja 1x1
            self.blocks = list(net.features)
        elif name == 'resnet50':
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, N_CLASSES))
            self.blocks = [nn.Sequential(net.conv1, net.bn1), net.layer1, net.layer2, net.layer3, net.layer4]
        else:
            raise ValueError(name)
        self.net = net
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.net(x)

    def cam_layer(self):
        return self.blocks[-1]

    def set_trainable(self, n_unfreeze):
        """Zamraża cały szkielet poza n ostatnimi blokami (głowica jest zawsze uczona)."""
        for p in self.net.parameters():
            p.requires_grad = False
        for block in self.blocks[len(self.blocks) - n_unfreeze:] if n_unfreeze > 0 else []:
            for p in block.parameters():
                p.requires_grad = True
        head = self.net.classifier if self.name == 'efficientnet_b0' else self.net.fc
        for p in head.parameters():
            p.requires_grad = True


def build_model(name, dropout=0.3):
    if name == 'cnn':
        return SimpleCNN(dropout)
    return Pretrained(name, dropout)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

import torch
from torch import nn
from torchvision import models


class ConvNet(nn.Module):
    def __init__(self, num_classes, pretrained: bool = True) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2]) # remove the last two layers (avgpool and fc) to get feature maps

        self.layer1 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(512, 64, 2),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True), # ceiling mode adds extra padding 
        )
        self.layer2 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 256, 2),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(256, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )
        self.layer3 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 32, 2),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(32, 16, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer4 = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # create a 1D vector for each image in the batch
        x = self.layer4(x)
        return x
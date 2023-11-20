import torch
import torchvision
import torch.nn as nn
from torch import Tensor

class ResNet50(nn.Module):
    def __init__(self, n_classes:int = 2) -> None:
        super().__init__()
        self.baseline = torchvision.models.resnet50(weights='DEFAULT')

        # Congelando o pesos de todas as camadas
        for param in self.baseline.parameters():
            param.requires_grad = False

        self.baseline.conv1.padding = (1,1)
        # Apenas a última camada FC terá seus pesos atualizados no treinamento
        self.baseline.fc = nn.Linear(2048, n_classes)

    def forward(self, x:Tensor) -> Tensor:
        x = self.baseline.conv1(x)
        x = self.baseline.bn1(x)
        x = self.baseline.relu(x)
        x = self.baseline.maxpool(x)

        x = self.baseline.layer1(x)
        x = self.baseline.layer2(x)
        x = self.baseline.layer3(x)
        x = self.baseline.layer4(x)

        x = self.baseline.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.baseline.fc(x)

        return x

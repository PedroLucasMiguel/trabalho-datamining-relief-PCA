import torch
import torchvision
import torch.nn as nn
from torch import Tensor

class EfficientNetB0(nn.Module):
    def __init__(self, n_classes:int = 2) -> None:
        super().__init__()
        self.baseline =torchvision.models.efficientnet_b0(weights='DEFAULT')

        # Congelando o pesos de todas as camadas
        for param in self.baseline.parameters():
            param.requires_grad = False

        self.baseline.classifier[1] = nn.Linear(1280, n_classes)

    def forward(self, x:Tensor) -> Tensor:
        x = self.baseline.features(x)
        x = self.baseline.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.baseline.classifier(x)

        return x

if __name__ == "__main__":
    print(EfficientNetB0(2))
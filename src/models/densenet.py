import torchvision
import torch.functional as F
import torch.nn as nn
from torch import flatten
from torch import Tensor

class DenseNet201(nn.Module):
    def __init__(self, n_classes:int = 2) -> None:
        super().__init__()
        self.baseline = torchvision.models.densenet201(weights='DEFAULT')
        #self.baseline = nn.Sequential(*(list(self.baseline.children())[:-1]))
    
        # Congelando o pesos de todas as camadas
        for param in self.baseline.parameters():
            param.requires_grad = False

        # Adicionando ultima camada de relu, avg_pool e ajustadno uma camada FC
        self.baseline.relu = nn.ReLU(inplace=True)
        self.baseline.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.baseline.classifier = nn.Linear(1920, n_classes)

    def forward(self, x:Tensor) -> Tensor:
        x = self.baseline.features(x)
        x = self.baseline.relu(x)
        x = self.baseline.avgpool(x)
        x = flatten(x, 1)
        x = self.baseline.classifier(x)

        return x


if __name__ == "__main__":
    print(DenseNet201(2))

import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn as nn
import torch

class Coral(ResNet):
    def __init__(self, num_classes=70):
        super(Coral, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes-1)
        self.fc= nn.Linear(2048, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(num_classes-1).float())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


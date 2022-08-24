import timm
import torch
from torch import nn

class ResNext(nn.Module):
    def __init__(self, num_classes: int, model_name = None, pretrained = True):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained)
        num_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-1]
        self.classify = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )

    def freeze_header(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_header(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, image):
        x = self.backbone(image)
        x = self.classify(x)
        return x

def resnext50_32x4d(num_classes: int, pretrained=True):
    if pretrained:
        print('Using pretrained model!')
    model = ResNext(num_classes, 'resnext50_32x4d', pretrained)
    return model
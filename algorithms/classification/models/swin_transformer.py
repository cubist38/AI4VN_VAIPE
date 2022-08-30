import timm
import torch
from torch import nn

class SwinTransformer(nn.Module):
    def __init__(self, num_classes: int, model_name = None, pretrained = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes = 0)
        num_features = self.backbone.num_features
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

def swin_tiny_transformer(num_classes: int, pretrained=True):
    if pretrained:
        print('Using pretrained model!')
    model = SwinTransformer(num_classes, 'swin_tiny_patch4_window7_224', pretrained)
    return model

def swin_small_transformer(num_classes: int, pretrained=True):
    if pretrained:
        print('Using pretrained model!')
    model = SwinTransformer(num_classes, 'swin_small_patch4_window7_224', pretrained)
    return model

def swin_base_transformer_384(num_classes: int, pretrained=True):
    if pretrained:
        print('Using pretrained model!')
    model = SwinTransformer(num_classes, 'swin_base_patch4_window12_384', pretrained)
    return model

def swin_large_transformer(num_classes: int, pretrained=True):
    if pretrained:
        print('Using pretrained model!')
    model = SwinTransformer(num_classes, 'swin_large_patch4_window7_224', pretrained)
    return model

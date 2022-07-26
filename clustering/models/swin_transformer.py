import timm
import torch
from torch import nn

class SwinTransformer(nn.Module):
    def __init__(self, model_name = None, pretrained = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes = 0)

    def freeze_header(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_header(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, image):
        x = self.backbone(image)
        return x

def swin_tiny_transformer(pretrained=True):
    if pretrained:
        print('Using pretrained model!')
    model = SwinTransformer('swin_tiny_patch4_window7_224', pretrained)
    return model

def swin_large_transformer(pretrained=True):
    if pretrained:
        print('Using pretrained model!')
    model = SwinTransformer('swin_large_patch4_window7_224', pretrained)
    return model
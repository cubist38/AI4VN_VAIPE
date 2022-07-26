from torch import nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small

class _Identity(nn.Module):
    '''
        This class is used when we don't want to calculate anything in classifier layers
    '''
    def __init__(self):
        super(_Identity, self).__init__()
        
    def forward(self, x):
        return x

class MobilenetV3(nn.Module):
    def __init__(self, option='small', pretrained=False):
        super(MobilenetV3, self).__init__()
        self.model = mobilenet_v3_large(pretrained=pretrained) if option != 'small' else mobilenet_v3_small(pretrained=pretrained)
        self.model.classifier = self.model.classifier[:2] # just using one Linear Layer + Hardswish in model classifier

    def forward(self, input):
        self.model
        return self.model(input)


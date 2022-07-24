import torch
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models.resnet import resnet18

class _Identity(torch.nn.Module):
    def __init__(self):
        super(_Identity, self).__init__()
        
    def forward(self, x):
        return x

class MobilenetV3():
    def __init__(self, option='small', pretrained=False):
        self.model = mobilenet_v3_large(pretrained=pretrained) if option != 'small' else mobilenet_v3_small(pretrained=pretrained)
        self.model.classifier = self.model.classifier[:2] # just using a Linear layer and its activation layer in model classifier
        # self.model.classifier = _Identity # using when dont wanna using classifier layers

    def predict(self, input):
        return self.model(input)

class Resnet18():
    def __init__(self, pretrained=False):
        self.model = resnet18(pretrained=pretrained)
        # self.model.fc = _Identity() # using when dont wanna using classifier layers
        
    def predict(self, input):
        return self.model(input)


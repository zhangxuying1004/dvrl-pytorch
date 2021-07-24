from torch import nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, architecture='inception_v3', downsample_factor=8, input_shape=(299, 299)):
        super(Encoder, self).__init__()
        self.architecture = architecture
        self.downsample_factor = downsample_factor
        self.input_shape = input_shape

        if architecture == 'inception_v3':
            self.base_model = models.inception_v3(pretrained=True)
        self.encoder = nn.Sequential(
            *list(self.base_model.children())[:-1],
            nn.AdaptiveAvgPool2d(self.downsample_factor, self.downsample_factor)
        )

        for param in self.encoder.parameters():
            param.required_grad = False

    def forward(self, x):
        x = x.resize(self.input_shape)
        out = self.encoder(x)
        return out
    
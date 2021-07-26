import torch
from torch import nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, architecture='inception_v3', downsample_factor=8, input_shape=(299, 299)):
        super(Encoder, self).__init__()
        self.architecture = architecture
        self.downsample_factor = downsample_factor
        self.input_shape = input_shape

        if architecture == 'inception_v3':
            # self.base_model = models.inception_v3(pretrained=False)
            # self.base_model = models.resnet18(pretrained=False)
            self.base_model = models.resnet50(pretrained=False)
            # self.base_model.load_state_dict(torch.load('./data_files/model_cache/inception_v3_google-1a9a5a14.pth'))
        self.encoder = nn.Sequential(
            *list(self.base_model.children())[:-2],
            nn.AvgPool2d((self.downsample_factor, self.downsample_factor), stride=8)
        )

        for param in self.encoder.parameters():
            param.required_grad = False

    def forward(self, x):
        # print('x before: ', x.shape)
        x = x.resize_(x.shape[0], x.shape[1], 299, 299)
        # print('x after: ', x.shape)
        out = self.encoder(x)
        return out
    
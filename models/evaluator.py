from torch import nn
from models.encoder import Encoder


class Evaluator(nn.Module):
    def __init__(self, feat_dim=2048, category_num=10):
        super(Evaluator, self).__init__()
        self.feat_dim = feat_dim
        self.category_num = category_num

        self.image_encoder = Encoder()
        self.layers = nn.Sequential(
            nn.Linear(self.feat_dim, self.category_num)
        )
    
    def forward(self, x):
        """
        x: (batch, 3, 32, 32)
        """
        x = self.image_encoder(x)
        assert x.shape[0] == self.feat_dim
    
        out = self.layers(x)
        return out



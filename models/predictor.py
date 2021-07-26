from torch import nn
from models.encoder import Encoder


class Predictor(nn.Module):
    def __init__(self, feat_dim=2048, category_num=10):
        super(Predictor, self).__init__()
        self.feat_dim = feat_dim
        self.category_num = category_num
        self.image_encoder = Encoder()
        self.pred_model = nn.Linear(self.feat_dim, self.category_num)

    def forward(self, x):
        """
        x: (batch, 3, 32, 32)
        """
        x = self.image_encoder(x)   # (batch, 3, 32, 32) => (batch, 2048)
        assert x.shape[-1] == self.feat_dim
        logits = self.pred_model(x)     # (batch, 2048) => (batch, category_num)
        return logits

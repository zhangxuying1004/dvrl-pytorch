from torch import nn


class Predictor(nn.Module):
    def __init__(self, feat_dim=2048, category_num=10):
        super(Predictor, self).__init__()
        self.feat_dim = feat_dim
        self.category_num = category_num
        self.pred_model = nn.Linear(self.feat_dim, self.category_num)

    def forward(self, x):
        logits = self.pred_model(x)
        return logits

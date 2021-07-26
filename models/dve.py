import torch
import torch.nn as nn
from utils import one_hot
from models.encoder import Encoder


class DVE(nn.Module):
    def __init__(self, feat_dim=2048, label_dim=10, hidden_dim=100, comb_dim=10, layer_num=5):
        """
        feat_dim: the dim of sample feature
        label_dim: the number of categories
        hidden_dim: the dim of hidden layers
        comb_dim: the dim of combine layer
        layer_num: the number of dve network
        """
        super(DVE, self).__init__()
        # 参数
        self.feat_dim = feat_dim
        self.label_dim = label_dim
        self.init_input_dim = feat_dim + label_dim
        self.hidden_dim = hidden_dim

        self.comb_input_dim = hidden_dim + label_dim
        self.comb_dim = comb_dim
        
        self.layer_num = layer_num
        
        # 模型层
        self.image_encoder = Encoder()  # 图像编码
        self.input_layer = nn.Sequential(
            nn.Linear(self.init_input_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.inter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim) ,
                nn.ReLU()
            ) for _ in range(self.layer_num-3)
        ])
        self.comb_layer = nn.Sequential(
            nn.Linear(self.comb_input_dim, self.comb_dim),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.comb_dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, y, y_hat):
        """
        x: (batch, 3, 32, 32), a batch of images
        y: (batch, ), ground-truth
        y_hat: (batch, label_dim), the difference between prediction and ground-truth
        """
        assert y_hat.shape[-1] == self.label_dim
        x = self.image_encoder(x)   # (batch, 3, 32, 32) => (batch, feat_dim)
        assert x.shape[-1] == self.feat_dim
        
        y = one_hot(y)  # (batch, ) => (batch, label_dim)
        init_input = torch.cat((x, y), dim=1)   # (batch, feat_dim+label_dim)
        init_output = self.input_layer(init_input)  # (batch, hidden_dim)

        inter_result = init_output
        for layer in self.inter_layers:
            inter_result = layer(inter_result)   # (batch, hidden_dim)

        comb_input = torch.cat((inter_result, y_hat), dim=1)   # (batch, hidden_dim+label_dim)
        comb_output = self.comb_layer(comb_input)   # (batch, comb_dim)

        dve = self.output_layer(comb_output)    # (batch,)
    
        return dve


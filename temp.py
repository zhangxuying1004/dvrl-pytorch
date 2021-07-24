from models.dve import DVE
from models.predictor import Predictor
import torch
import numpy as np

def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth).long()
    idx = torch.unsqueeze(label, dim=1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

def test():
    a = torch.from_numpy(np.array([7, 3, 8, 6, 7, 8, 9, 2]))
    print(a.shape, a)
    b = one_hot(a)
    print(b.shape, b)

def test_dve():
    batch_size = 64
    feat_dim, label_dim = 2048, 10

    dve_model = DVE(feat_dim, label_dim)
    print(dve_model)

    x = torch.randn(batch_size, feat_dim)
    y = torch.randn(batch_size, label_dim)
    y_hat = torch.randn(batch_size, label_dim)

    print(x.shape)
    print(y.shape)
    print(y_hat.shape)

    output = dve_model(x, y, y_hat)
    print(output.shape)


def test_pred():
    batch_size = 64
    feat_dim = 2048 
    x = torch.randn(batch_size, feat_dim)
    pred_model = Predictor()
    output = pred_model(x)
    print(output.shape)

test()

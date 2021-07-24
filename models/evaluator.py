from torch import nn


class Evaluator(nn.Module):
    def __init__(self, class_num=10):
        super(Evaluator, self).__init__()
        self.class_num = class_num

        self.model = nn.Sequential(
            nn.Linear(2048, self.class_num)
        )
    
    def forward(self, x):
        out = self.model(x)
        return out



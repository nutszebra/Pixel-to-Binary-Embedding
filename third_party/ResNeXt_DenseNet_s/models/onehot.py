import torch
import torch.nn as nn


class Onehot(nn.Module):

    def __init__(self, m=32):
        super(Onehot, self).__init__()
        self.m = m
        embedding = torch.zeros(m, m)
        for i in range(m):
            embedding[i, i] = 1.0

        self.register_buffer('embedding', embedding)

    def forward(self, x):
        b, c, h, w = x.shape
        indices = torch.clamp(x * self.m, 0, self.m - 1)
        return self.embedding[(indices).long().view(-1)].reshape(b, c, h, w, self.embedding.shape[1]).permute(0, 1, -1, 2, 3).reshape(b, -1, h, w)


if __name__ == '__main__':
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    onehot = Onehot(m=4)
    y = onehot(x)

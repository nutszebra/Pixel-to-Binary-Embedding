import torch
import torch.nn as nn

try:
    from .ste import ste
except:
    from ste import ste


class P2BE(nn.Module):

    def __init__(self, m=32, beta=-1.0e-1, init='normal'):
        super(P2BE, self).__init__()
        self.m, self.beta = m, beta
        if init == 'normal':
            embedding = torch.randn(256, m)
        elif init == 'thermometer':
            embedding = torch.zeros(m, m)
            for i in range(m):
                embedding[i, i:] = 1.0
            embedding = torch.nn.functional.interpolate(embedding.view(1, 1, m, m), (256, m), mode='nearest')[0, 0] - 0.5
        elif init == 'onehot':
            embedding = torch.zeros(m, m)
            for i in range(m):
                embedding[i, i] = 1.0
            embedding = torch.nn.functional.interpolate(embedding.view(1, 1, m, m), (256, m), mode='nearest')[0, 0] - 0.5

        self.register_parameter('embedding', torch.nn.Parameter(embedding))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.embedding[(x * 255).long().view(-1)].reshape(b, c, h, w, self.embedding.shape[1]).permute(0, 1, -1, 2, 3).reshape(b, -1, h, w)
        # x = ste(x, binary=True)
        x = (ste(x, binary=True) + 1) / 2.0
        return x

    def embedding_smoothness(self):
        def cosine(z1, z2, eps=1.0e-8):
            return ((z1 * z2) / (torch.norm(z1, dim=1) * torch.norm(z2, dim=1) + eps).unsqueeze(1)).sum(1)
        similarity = cosine(self.embedding[:-1], self.embedding[1:]).mean()
        return self.beta * similarity


if __name__ == '__main__':
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    p2be = P2BE(m=4)
    y = p2be(x)

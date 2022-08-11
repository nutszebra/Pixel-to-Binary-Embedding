import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction


class STE(InplaceFunction):

    @staticmethod
    def forward(ctx, input1, binary=True):
        ctx.input1 = input1
        if binary is True:
            out = (input1 >= 0).float() - (input1 < 0).float()
        else:
            d1 = (input1 <= -1).float()
            d2 = ((input1 > -1) * (input1 <= 0)).float()
            d3 = ((input1 >= 0) * (input1 < 1)).float()
            d4 = (input1 >= 1).float()
            out = - d1
            out = out + (2.0 * ctx.input1 + ctx.input1 ** 2) * d2
            out = out + (2.0 * ctx.input1 - ctx.input1 ** 2) * d3
            out = out + d4
        return out

    @staticmethod
    def backward(ctx, grad_output):
        d2 = ((ctx.input1 > -1) * (ctx.input1 <= 0)).float()
        d3 = ((ctx.input1 >= 0) * (ctx.input1 < 1)).float()

        d = (2.0 + 2 * ctx.input1) * d2
        d = d + (2.0 - 2 * ctx.input1) * d3

        return grad_output * d, None


def ste(input1, binary=True):
    return STE.apply(input1, binary)


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


class LSPGA(nn.Module):

    def __init__(self, model, encoder, epsilon=8. / 255., delta=1.2, xi=1.5, step=7, criterion=nn.CrossEntropyLoss(), p2be=False):
        super(LSPGA, self).__init__()
        self.model = model
        self.epsilon = epsilon        # l-inf norm bound of image
        self.delta = delta            # annealing factor
        self.xi = xi                  # step-size of attack
        self.step = step              # attack steps
        self.criterion = criterion    # loss func
        self.encoder = encoder        # binary encoder
        self.onehot = Onehot(m=self.encoder.embedding.shape[0])     # onehot encoder
        self.p2be = p2be

    def getmask(self, x):
        b, c, h, w = x.shape
        mask = x.new_zeros((b, int(c * self.encoder.embedding.shape[0]), h, w))
        low = x - self.epsilon
        low[low < 0] = 0
        high = x + self.epsilon
        high[high > 1] = 1
        for i in range(self.encoder.embedding.shape[0] + 1):
            interimg = (i * 1. / self.encoder.embedding.shape[0]) * low + (1 - i * 1. / self.encoder.embedding.shape[0]) * high
            mask += self.onehot(interimg)
        mask[mask > 1] = 1
        return mask

    def forward(self, data, target):
        b, c, h, w = data.shape
        mask = self.getmask(data)
        u = torch.normal(0, 1, mask.shape, device=self.encoder.embedding.device) - (1.0 - mask) * 1.0e10
        u = u.requires_grad_(True)
        T = 1.0

        # no grad for memory usage
        with torch.no_grad():
            if self.p2be is False:
                embedding = self.encoder.embedding
            else:
                # embedding = ste(self.encoder.embedding, binary=True)
                embedding = (ste(self.encoder.embedding, binary=True) + 1.0) / 2.0

        z = (embedding.view(1, 1, embedding.shape[0], 1, 1, embedding.shape[1]) * F.softmax(u.view(b, -1, embedding.shape[0], h, w) / T, dim=2).unsqueeze(-1)).sum(2).permute(0, 1, -1, 2, 3).reshape(b, -1, h, w)

        for t in range(self.step):
            out = self.model(z, binary_embedding=False)
            loss = self.criterion(out, target)
            u_grad = torch.autograd.grad(loss, u)[0]
            u = (self.xi * torch.sign(u_grad) + u)
            z = (embedding.view(1, 1, embedding.shape[0], 1, 1, embedding.shape[1]) * F.softmax(u.view(b, -1, embedding.shape[0], h, w) / T, dim=2).unsqueeze(-1)).sum(2).permute(0, 1, -1, 2, 3).reshape(b, -1, h, w)
            T = T * self.delta
            del out
            del loss
            del u_grad
        del z
        u = u.detach()
        adversarial_indices = u.view(b, -1, embedding.shape[0], h, w).argmax(2, keepdim=False).long().view(-1)
        z_onehot = self.encoder.embedding[adversarial_indices].reshape(b, c, h, w, embedding.shape[1]).permute(0, 1, -1, 2, 3).reshape(b, -1, h, w)

        if self.p2be is False:
            pass
        else:
            # z_onehot = ste(z_onehot, binary=True)
            z_onehot = (ste(z_onehot, binary=True) + 1.0) / 2.0

        return z_onehot

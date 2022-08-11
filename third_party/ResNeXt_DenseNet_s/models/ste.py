import torch
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


if __name__ == '__main__':
    import torch
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    y_b = ste(x, True)
    y = ste(x, False)

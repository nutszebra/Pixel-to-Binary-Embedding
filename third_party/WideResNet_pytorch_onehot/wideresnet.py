"""WideResNet implementation (https://arxiv.org/abs/1605.07146)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .color_transformations import SRGB2XYZ, XYZ2MHPT, XYZ2MCAT02, YIQ2SRGB, SRGB2YIQ
    from .convlstm import ConvLSTMCell
    from .convgru import ConvGRUCell
except:
    from color_transformations import SRGB2XYZ, XYZ2MHPT, XYZ2MCAT02, YIQ2SRGB, SRGB2YIQ
    from convlstm import ConvLSTMCell
    from convgru import ConvGRUCell

# torch.backends.cudnn.enabled = False


class BasicBlock(nn.Module):
  """Basic ResNet block."""

  def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
    super(BasicBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)
    self.bn2 = nn.BatchNorm2d(out_planes)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(
        out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.drop_rate = drop_rate
    self.is_in_equal_out = (in_planes == out_planes)
    self.conv_shortcut = (not self.is_in_equal_out) and nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False) or None

  def forward(self, x):
    if not self.is_in_equal_out:
      x = self.relu1(self.bn1(x))
    else:
      out = self.relu1(self.bn1(x))
    if self.is_in_equal_out:
      out = self.relu2(self.bn2(self.conv1(out)))
    else:
      out = self.relu2(self.bn2(self.conv1(x)))
    if self.drop_rate > 0:
      out = F.dropout(out, p=self.drop_rate, training=self.training)
    out = self.conv2(out)
    if not self.is_in_equal_out:
      return torch.add(self.conv_shortcut(x), out)
    else:
      return torch.add(x, out)


class NetworkBlock(nn.Module):
  """Layer container for blocks."""

  def __init__(self,
               nb_layers,
               in_planes,
               out_planes,
               block,
               stride,
               drop_rate=0.0):
    super(NetworkBlock, self).__init__()
    self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                  stride, drop_rate)

  def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                  drop_rate):
    layers = []
    for i in range(nb_layers):
      layers.append(
          block(i == 0 and in_planes or out_planes, out_planes,
                i == 0 and stride or 1, drop_rate))
    return nn.Sequential(*layers)

  def forward(self, x):
    return self.layer(x)


class WideResNetS(nn.Module):
  """WideResNet class."""

  def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0, train_m=1, train_n=1000, bptt_rate=0.005, truncate=20, test_m=1, test_n=1000, m=16):
    super(WideResNetS, self).__init__()
    self.train_m, self.test_m= train_m, test_m
    self.train_n, self.test_n = train_n, test_n
    self.truncate, self.bptt_rate = truncate, bptt_rate
    self.toxyz = SRGB2XYZ()
    self.tomcat02 = XYZ2MCAT02()
    self.fromyiqtorgb = YIQ2SRGB()
    self.fromrgbtoyiq = SRGB2YIQ()

    n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    assert (depth - 4) % 6 == 0
    n = (depth - 4) // 6
    block = BasicBlock
    # 1st conv before any network block
    self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
    # self.conv1 = nn.Conv2d(
    #     3 * self.train_n, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
    self.lstm1 = ConvLSTMCell(in_channels=3, hidden_channels=m, kernel_size=(1, 1), bias=True)
    self.lstm2 = ConvLSTMCell(in_channels=m, hidden_channels=3, kernel_size=(1, 1), bias=True)
    # self.gru1 = ConvGRUCell(in_channels=3, hidden_channels=m, kernel_size=(1, 1), bias=True)
    # self.gru2 = ConvGRUCell(in_channels=m, hidden_channels=3, kernel_size=(1, 1), bias=True)
    # 1st block
    self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1,
                               drop_rate)
    # 2nd block
    self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2,
                               drop_rate)
    # 3rd block
    self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2,
                               drop_rate)
    # global average pooling and classifier
    self.bn1 = nn.BatchNorm2d(n_channels[3])
    self.relu = nn.ReLU(inplace=True)
    self.fc = nn.Linear(n_channels[3], num_classes)
    self.n_channels = n_channels[3]

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

  @staticmethod
  def flip(x, dim):
      indices = [slice(None)] * x.dim()
      indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                  dtype=torch.long, device=x.device)
      return x[tuple(indices)]

  def forward(self, x):
    if self.training is True:
      # lstm
      b, _, height, width = x.shape
      h1 = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      c1 = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      h2 = x.new_zeros(b, self.lstm2.hidden_dim, height, width)
      c2 = x.new_zeros(b, self.lstm2.hidden_dim, height, width)
      x_s = torch.distributions.Bernoulli(x).sample((self.train_n, self.train_m)).mean(1).transpose(0, 1)
      tmp = 0.0
      counter = 0
      bptt_indices = np.random.permutation(self.train_n)[:int(self.train_n * self.bptt_rate)].tolist()
      for i in range(self.train_n):
          h1, c1 = self.lstm1(input_tensor=x_s[:, i], h_cur=h1, c_cur=c1)
          h2, c2 = self.lstm2(input_tensor=h1, h_cur=h2, c_cur=c2)
          if counter <= self.truncate:
              counter += 1
          else:
              h1, c1 = h1.detach(), c1.detach()
              h2, c2 = h2.detach(), c2.detach()
              counter = 0
          if i in bptt_indices:
              tmp = tmp + h2
          else:
              tmp = tmp + h2.detach()
      x = tmp / self.train_n
      # gru 
      # b, _, height, width = x.shape
      # h1 = x.new_zeros(b, self.gru1.hidden_dim, height, width)
      # h2 = x.new_zeros(b, self.gru2.hidden_dim, height, width)
      # x_s = torch.distributions.Bernoulli(x).sample((self.train_n, self.train_m)).mean(1).transpose(0, 1)
      # tmp = 0.0
      # for i in range(self.train_n):
      #     h1 = self.gru1(input_tensor=x_s[:, i], h_cur=h1)
      #     h2 = self.gru2(input_tensor=h1, h_cur=h2)
      #     tmp = tmp + h2
      # x = tmp / self.train_n
      # pass
      # h1 = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      # c1 = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      # h2 = x.new_zeros(b, self.lstm2.hidden_dim, height, width)
      # c2 = x.new_zeros(b, self.lstm2.hidden_dim, height, width)
      # tmp = 0.0
      # for i in range(int(self.train_length / self.train_n)):
      #     x_s = torch.distributions.Bernoulli(x).sample((self.train_n, )).transpose(0, 1)
      #     b, _, _, height, width = x_s.shape
      #     h1, c1, h2, c2 = h1.detach(), c1.detach(), h2.detach(), c2.detach()
      #     for i in range(self.train_n):
      #         h1, c1 = self.lstm1(input_tensor=x_s[:, i], h_cur=h1, c_cur=c1)
      #         h2, c2 = self.lstm2(input_tensor=h1, h_cur=h2, c_cur=c2)
      #     tmp = tmp + (self.train_n / self.train_length) * h2
      # x = tmp
      # h = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      # c = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      # for i in range(self.train_n):
      #     h, c = self.lstm1(input_tensor=x_s[:, i].clone(),
      #                       h_cur=h, c_cur=c)
      #     h, c = h.clone(), c.clone()
      # out = h.clone()

      # x_ref = self.fromrgbtoyiq(x)
      # x_ref[:, 1:] = self.flip(x_ref, 0)[:, 1:]
      # x_ref = self.fromyiqtorgb(x_ref).clamp(0.0, 1.0)
      # beta = 1.0 - torch.rand(x.shape[0], device=x.device).reshape(-1, 1, 1, 1) * 0.5
      # x = beta * x + (1.0 - beta) * x_ref
      # pass
      # x = x / x.view(x.shape[0], -1).max(axis=1)[0].view(-1, 1, 1, 1)
      # X = []
      # for i in range(1, self.train_n + 1):
      #   X.append((x >= (i / (self.train_n + 1))).float().unsqueeze(0))
      # x = torch.cat(X, 0).transpose(0, 1).reshape(x.shape[0], -1, x.shape[2], x.shape[3])

      # i = self.train_n
      # i = self.train_n[np.random.randint(0, len(self.train_n))]
      # RGB
      # x = torch.distributions.Poisson(x).sample((i, )).mean(0)
      # Normal
      # i = np.random.randint(1, self.train_n)
      # x = torch.distributions.Normal(x, 0.2).sample((i, )).mean(0)
      # LMS
      # Bernoulli
      # x = torch.distributions.Bernoulli(torch.clamp(self.tomcat02(self.toxyz(x)), 0.0, 1.0)).sample((i, )).mean(0)
      # Poisson 
      # x = torch.distributions.Poisson(self.tomcat02(self.toxyz(x))).sample((i, )).mean(0)
      # mixing
      # alpha = np.random.rand()
      # x_s = torch.distributions.Poisson(self.tomcat02(self.toxyz(x))).sample((i, )).mean(0)
      # x = alpha * x + (1.0 - alpha) * x_s
    else:
      # lstm
      b, _, height, width = x.shape
      h1 = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      c1 = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      h2 = x.new_zeros(b, self.lstm2.hidden_dim, height, width)
      c2 = x.new_zeros(b, self.lstm2.hidden_dim, height, width)
      x_s = torch.distributions.Bernoulli(x).sample((self.test_n, self.test_m)).mean(1).transpose(0, 1)
      tmp = 0.0
      for i in range(self.test_n):
          h1, c1 = self.lstm1(input_tensor=x_s[:, i], h_cur=h1, c_cur=c1)
          h2, c2 = self.lstm2(input_tensor=h1, h_cur=h2, c_cur=c2)
          tmp = tmp + h2
      x = tmp / self.test_n

      # gru 
      # b, _, height, width = x.shape
      # h1 = x.new_zeros(b, self.gru1.hidden_dim, height, width)
      # h2 = x.new_zeros(b, self.gru2.hidden_dim, height, width)
      # x_s = torch.distributions.Bernoulli(x).sample((self.test_n, self.test_m)).mean(1).transpose(0, 1)
      # tmp = 0.0
      # for i in range(self.test_n):
      #     h1 = self.gru1(input_tensor=x_s[:, i], h_cur=h1)
      #     h2 = self.gru2(input_tensor=h1, h_cur=h2)
      #     tmp = tmp + h2
      # x = tmp / self.test_n

      # h1 = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      # c1 = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      # h2 = x.new_zeros(b, self.lstm2.hidden_dim, height, width)
      # c2 = x.new_zeros(b, self.lstm2.hidden_dim, height, width)
      # tmp = 0.0
      # for i in range(int(self.test_length / self.test_n)):
      #     x_s = torch.distributions.Bernoulli(x).sample((self.test_n, )).transpose(0, 1)
      #     b, _, _, height, width = x_s.shape
      #     h1, c1, h2, c2 = h1.detach(), c1.detach(), h2.detach(), c2.detach()
      #     for i in range(self.test_n):
      #         h1, c1 = self.lstm1(input_tensor=x_s[:, i], h_cur=h1, c_cur=c1)
      #         h2, c2 = self.lstm2(input_tensor=h1, h_cur=h2, c_cur=c2)
      #     tmp = tmp + (self.test_n / self.test_n) * h2
      # x = tmp
      # h = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      # c = x.new_zeros(b, self.lstm1.hidden_dim, height, width)
      # for i in range(self.test_n):
      #     h, c = self.lstm1(input_tensor=x_s[:, i].clone(),
      #                       h_cur=h, c_cur=c)
      #     h, c = h.clone(), c.clone()
      # out = h

      # x = x / x.view(x.shape[0], -1).max(axis=1)[0].view(-1, 1, 1, 1)
      # X = []
      # for i in range(1, self.train_n + 1):
      #   X.append((x >= (i / (self.train_n + 1))).float().unsqueeze(0))
      # x = torch.cat(X, 0).transpose(0, 1).reshape(x.shape[0], -1, x.shape[2], x.shape[3])

      # i = self.test_n
      # RGB
      # x = torch.distributions.Poisson(x).sample((i, )).mean(0)
      # Normal
      # x = torch.distributions.Normal(x, 0.2).sample((i, )).mean(0)
      # LMS
      # Bernoulli
      # x = torch.distributions.Bernoulli(torch.clamp(self.tomcat02(self.toxyz(x)), 0.0, 1.0)).sample((i, )).mean(0)
      # Poisson 
      # x = torch.distributions.Poisson(self.tomcat02(self.toxyz(x))).sample((i, )).mean(0)
    out = self.conv1(x)
    out = self.block1(out)
    out = self.block2(out)
    out = self.block3(out)
    out = self.relu(self.bn1(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(-1, self.n_channels)
    return self.fc(out)

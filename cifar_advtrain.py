"""Main script to launch adversarial training on CIFAR-10/100.

Supports WideResNet, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on adversarially perturbed CIFAR-10-C and CIFAR-100-C.

Four types of input space: RGB, one-hot encoding, thermometer encoding, and P2BE.
"""
from __future__ import print_function

import argparse
import os
import shutil
import time

import numpy as np
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.ResNeXt_DenseNet_s.models.densenet_b import densenetb
from third_party.ResNeXt_DenseNet_s.models.resnext_b import resnext29b
from third_party.ResNeXt_DenseNet_s.models.densenet_onehot import densenet_onehot
from third_party.ResNeXt_DenseNet_s.models.densenet_thermometer import densenet_thermometer
from third_party.ResNeXt_DenseNet_s.models.resnext_onehot import resnext29_onehot
from third_party.ResNeXt_DenseNet_s.models.resnext_thermometer import resnext29_thermometer
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from third_party.WideResNet_pytorch_s.wideresnet_b import WideResNetB
from third_party.WideResNet_pytorch_onehot.wideresnet_onehot import WideResNetOnehot
from third_party.WideResNet_pytorch_thermometer.wideresnet_thermometer import WideResNetThermometer

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from lspga import LSPGA

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='wrn',
    choices=['wrn', 'wrnb', 'wrnonehot', 'wrnthermo', 'densenet', 'densenetb', 'densenetonehot', 'densenetthermo', 'resnext', 'resnextb', 'resnextonehot', 'resnextthermo'],
    help='Choose architecture.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument(
    '--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.0, type=float, help='Dropout probability')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
parser.add_argument(
    '--split',
    type=int,
    default=16,
    help='splitted process for generating adv')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')
# pixel embedding paramters
parser.add_argument(
    '--m',
    type=int,
    default=32,
    help='dimension of pixel embedding.')
parser.add_argument(
    '--coefficient_smooth',
    type=float,
    default=1.0e-1,
    help='the coefficient of embedding smoothness loss')
parser.add_argument(
    '--wde',
    type=float,
    default=1.0e-4,
    help='Weight decay (L2 penalty) on embedding.')
parser.add_argument(
    '--lre',
    type=float,
    default=1.0e-4,
    help='Weight decay (L2 penalty) on embedding.')
parser.add_argument(
    '--xi',
    type=float,
    default=1.5,
    help='stepsize of lspga')
parser.add_argument(
    '--mode',
    type=str,
    default='consistency',
    choices=['clean', 'onlyadv', 'consistency'],
    help='Choose Training Method.')
parser.add_argument(
    '--p2be_init',
    type=str,
    default='normal',
    choices=['normal', 'onehot', 'thermometer'],
    help='Choose Initializing Method for P2BE.')

args = parser.parse_args()


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def train(net, train_loader, optimizer, scheduler, model_name=args.model, optimizer_e=None, scheduler_e=None, adversarial_training=True, xi=args.xi, split=args.split, mode=args.mode):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.

  for i, (images, targets) in enumerate(train_loader):

    images, targets = images.cuda(), targets.cuda()

    if 'onehot' in args.model:
      encoder = net._modules['module'].onehot
      p2be = False
    elif 'thermo' in args.model:
      encoder = net._modules['module'].thermometer
      p2be = False
    elif 'b' == args.model[-1]:
      encoder = net._modules['module'].p2be
      p2be = True

    if mode == 'clean':
      images = images.cuda()
      targets = targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
    else:
      images_adv = []
      adversary = LSPGA(net._modules['module'], encoder=encoder, p2be=p2be, xi=xi)
      adversary = torch.nn.DataParallel(adversary).cuda()
      net.eval()
      adversary.eval()
      for _x, _t in zip(torch.split(images, split, dim=0), torch.split(targets, split, dim=0)):
        images_adv.append(adversary(_x, _t))
      net.train()
      images_adv = torch.cat(images_adv, 0)

      if mode == 'onlyadv':
        logits = net(images_adv, binary_embedding=False)
        loss = F.cross_entropy(logits, targets)
      elif mode == 'consistency':
        logits = net(torch.cat((encoder(images), images_adv), 0), binary_embedding=False)
        logits_clean, logits_adv = torch.split(logits, images.shape[0])
        loss = F.cross_entropy(logits_clean, targets)
        p_clean, p_adv = F.softmax(logits_clean, dim=1), F.softmax(logits_adv, dim=1)
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_adv) / 2., 1e-7, 1).log()
        loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                      F.kl_div(p_mixture, p_adv, reduction='batchmean')) / 2.
      else:
        raise Exception('mode is {}'.format(mode))
    # embedding loss
    if model_name == 'wrnb' or model_name == 'densenetb' or model_name == 'resnextb':
      similarity = net._modules['module'].embedding_smoothness()
      loss = loss - similarity

    optimizer.zero_grad()
    if optimizer_e is not None:
      optimizer_e.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if optimizer_e is not None:
      optimizer_e.step()
    if scheduler_e is not None:
      scheduler_e.step()

    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    if i % args.print_freq == 0:
      print('Train Loss {:.3f}'.format(loss_ema))

  return loss_ema


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_adv(net, test_loader, xi=args.xi):

  net.eval()
  total_correct = 0

  if 'onehot' in args.model:
    encoder = net._modules['module'].onehot
    p2be = False
  elif 'thermo' in args.model:
    encoder = net._modules['module'].thermometer
    p2be = False
  elif 'b' == args.model[-1]:
    encoder = net._modules['module'].p2be
    p2be = True

  adversary = LSPGA(net._modules['module'], encoder=encoder, p2be=p2be, xi=xi)
  adversary = torch.nn.DataParallel(adversary).cuda()

  for images, targets in test_loader:
    images, targets = images.cuda(), targets.cuda()
    images_adv = adversary(images, targets)
    with torch.no_grad():
      logits = net(images_adv, binary_embedding=False)
    loss = F.cross_entropy(logits, targets)
    pred = logits.data.max(1)[1]
    total_correct += pred.eq(targets.data).sum().item()

  return total_correct / len(test_loader.dataset)


def main():
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       transforms.ToTensor()])

  preprocess = transforms.Compose([transforms.ToTensor()])
  test_transform = preprocess

  if args.dataset == 'cifar10':
    train_data = datasets.CIFAR10(
        './data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(
        './data/cifar', train=False, transform=test_transform, download=True)
    base_c_path = './data/cifar-c/CIFAR-10-C/'
    num_classes = 10
  else:
    train_data = datasets.CIFAR100(
        './data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100(
        './data/cifar', train=False, transform=test_transform, download=True)
    base_c_path = './data/cifar-c/CIFAR-100-C/'
    num_classes = 100

  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True)

  test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  # Create model
  if args.model == 'densenet':
    net = densenet(num_classes=num_classes)
  elif args.model == 'densenetonehot':
    print('{} m: {}'.format(args.model, args.m))
    net = densenet_onehot(num_classes=num_classes, m=args.m)
  elif args.model == 'densenetthermo':
    print('{} m: {}'.format(args.model, args.m))
    net = densenet_thermometer(num_classes=num_classes, m=args.m)
  elif args.model == 'densenetb':
    print('{} m: {}, lambda: {}'.format(args.model, args.m, args.coefficient_smooth))
    net = densenetb(num_classes=num_classes, m=args.m, beta=args.coefficient_smooth)
  elif args.model == 'wrn':
    net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
  elif args.model == 'wrnb':
    print('{} m: {}, lambda: {},  init: {}'.format(args.model, args.m, args.coefficient_smooth, args.p2be_init))
    net = WideResNetB(args.layers, num_classes, args.widen_factor, args.droprate, m=args.m, beta=args.coefficient_smooth, init=args.p2be_init)
  elif args.model == 'wrnonehot':
    print('{} m: {}'.format(args.model, args.m))
    net = WideResNetOnehot(args.layers, num_classes, args.widen_factor, args.droprate, m=args.m)
  elif args.model == 'wrnthermo':
    print('{} m: {}'.format(args.model, args.m))
    net = WideResNetThermometer(args.layers, num_classes, args.widen_factor, args.droprate, m=args.m)
  elif args.model == 'resnext':
    net = resnext29(num_classes=num_classes)
  elif args.model == 'resnextonehot':
    print('{} m: {}'.format(args.model, args.m))
    net = resnext29_onehot(num_classes=num_classes, m=args.m)
  elif args.model == 'resnextthermo':
    print('{} m: {}'.format(args.model, args.m))
    net = resnext29_thermometer(num_classes=num_classes, m=args.m)
  elif args.model == 'resnextb':
    print('{} m: {}, lambda: {}'.format(args.model, args.m, args.coefficient_smooth))
    net = resnext29b(num_classes=num_classes, m=args.m, beta=args.coefficient_smooth)

  if args.model == 'wrnb' or args.model == 'densenetb' or args.model == 'resnextb' or args.model == 'wrnblinear':
    optimizer = torch.optim.SGD([{'params': net.base_parameters()}],
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.decay,
            nesterov=True)
    print('optmizer e: {}'.format(args.lre, args.wde))
    optimizer_e = torch.optim.AdamW([{'params': net.embedding_parameters()}],
            args.lre,
            betas=(0.999, 0.999),
            weight_decay=args.wde)
  else:
    optimizer = torch.optim.SGD(
            net.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.decay,
            nesterov=True)
    optimizer_e = None

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0

  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('Model restored from epoch:', start_epoch)

  if args.evaluate:
    # Evaluate clean accuracy first because test_c mutates underlying data
    test_loss, test_acc = test(net, test_loader)
    print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
        test_loss, 100 - 100. * test_acc))

    test_adv_acc = test_adv(net, test_loader)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_adv_acc))
    return

  scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
      step,
      args.epochs * len(train_loader),
      1,  # lr_lambda computes multiplicative factor
      1e-6 / args.learning_rate))

  if args.model == 'wrnb' or args.model == 'densenetb' or args.model == 'resnextb':
    scheduler_e = torch.optim.lr_scheduler.LambdaLR(
      optimizer_e,
      lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
      step,
      args.epochs * len(train_loader),
      1,  # lr_lambda computes multiplicative factor
      1e-6 / args.lre))
  else:
    scheduler_e = None

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          args.dataset + '_' + args.model + '_training_log.csv')
  with open(log_path, 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

  best_acc = 0
  print('Beginning training from epoch:', start_epoch + 1)
  for epoch in range(start_epoch, args.epochs):
    begin_time = time.time()

    if epoch == start_epoch:
      train_loss_ema = train(net, train_loader, optimizer, scheduler, model_name=args.model, optimizer_e=optimizer_e, scheduler_e=scheduler_e, adversarial_training=False)
    else:
      train_loss_ema = train(net, train_loader, optimizer, scheduler, model_name=args.model, optimizer_e=optimizer_e, scheduler_e=scheduler_e, adversarial_training=True)
    test_loss, test_acc = test(net, test_loader)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          time.time() - begin_time,
          train_loss_ema,
          test_loss,
          100 - 100. * test_acc,
      ))

    print(
        'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
        ' Test Error {4:.2f}'
        .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                test_loss, 100 - 100. * test_acc))

  test_adv_acc = test_adv(net, test_loader)
  print('Mean Adversarial Error: {:.3f}'.format(100 - 100. * test_adv_acc))

  with open(log_path, 'a') as f:
    f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
            (args.epochs + 1, 0, 0, 0, 100 - 100 * test_adv_acc))


if __name__ == '__main__':
  main()

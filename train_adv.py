import os
import sys
import time
import glob
import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

import genotypes
import utils
from genotypes import Genotype
from model import NetworkCIFAR as Network


def get_cifar_dataset_class(dataset):
  dataset = dataset.lower()
  if dataset == 'cifar10':
    return dset.CIFAR10
  if dataset == 'cifar100':
    return dset.CIFAR100
  if dataset == 'svhn':
    return dset.SVHN
  raise ValueError('unsupported dataset: {}'.format(dataset))


def build_dataset(dataset, root, train, transform):
  DatasetClass = get_cifar_dataset_class(dataset)
  if dataset.lower() == 'svhn':
    split = 'train' if train else 'test'
    return DatasetClass(
        root=root,
        split=split,
        download=True,
        transform=transform)
  return DatasetClass(
      root=root,
      train=train,
      download=True,
      transform=transform)


parser = argparse.ArgumentParser("cifar_adv_train")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'], help='dataset to use')
parser.add_argument('--cifar_mean', type=float, nargs=3, default=None, help='optional RGB normalization mean')
parser.add_argument('--cifar_std', type=float, nargs=3, default=None, help='optional RGB normalization std')
parser.add_argument('--genotype_path', type=str, default=None, help='path to searched genotype.txt')
parser.add_argument('--arch', type=str, default=None, help='named genotype in genotypes.py, used if genotype_path is not set')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of adversarial training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='ADV-EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--num_workers', type=int, default=2, help='data loader workers')

parser.add_argument('--adv_eps', type=float, default=8 / 255, help='PGD training epsilon')
parser.add_argument('--adv_alpha', type=float, default=2 / 255, help='PGD training step size')
parser.add_argument('--adv_steps', type=int, default=7, help='PGD training steps')
parser.add_argument('--no_random_start', action='store_true', default=False, help='disable PGD random start')

parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[100, 150], help='step LR decay epochs')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='step LR decay factor')
parser.add_argument('--eval_freq', type=int, default=10, help='deprecated: use pgd_eval_freq instead')
parser.add_argument('--clean_eval_freq', type=int, default=1, help='run clean validation every n epochs; set <=0 to disable')
parser.add_argument('--pgd_eval_freq', type=int, default=None, help='run PGD validation every n epochs; defaults to eval_freq')
parser.add_argument('--eval_pgd_steps', type=int, default=20, help='PGD steps for periodic validation')
parser.add_argument('--eval_pgd100_final', action='store_true', default=False, help='also run PGD-100 at the final epoch')
parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')


class NormalizeWrapper(nn.Module):
  def __init__(self, model, mean=None, std=None):
    super(NormalizeWrapper, self).__init__()
    self.model = model
    mean, std = utils.get_cifar_mean_std('cifar10', mean=mean, std=std)
    self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
    self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

  def forward(self, x):
    x = (x - self.mean) / self.std
    return self.model(x)


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def load_genotype(path):
  with open(path, 'r') as f:
    genotype_str = f.read().strip()
  return eval(genotype_str, {"Genotype": Genotype, "range": range})


def logits_only(output):
  if isinstance(output, tuple):
    return output[0], output[1]
  return output, None


@torch.enable_grad()
def pgd_attack_train(model, images, labels, eps, alpha, steps, random_start=True):
  was_training = model.training
  model.eval()

  x_adv = images.detach().clone()
  if random_start:
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

  for _ in range(steps):
    x_adv.requires_grad_()
    logits, _ = logits_only(model(x_adv))
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

    x_adv = x_adv.detach() + alpha * grad.sign()
    x_adv = torch.max(torch.min(x_adv, images + eps), images - eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

  if was_training:
    model.train()
  else:
    model.eval()

  return x_adv.detach()


def pgd_attack_eval(model, images, labels, eps, alpha, steps, batch_id, seed):
  torch.manual_seed(seed + batch_id)
  torch.cuda.manual_seed_all(seed + batch_id)
  return pgd_attack_train(model, images, labels, eps, alpha, steps, random_start=True)


def build_loaders(args):
  train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
  ])
  valid_transform = transforms.Compose([
      transforms.ToTensor(),
  ])

  train_data = build_dataset(args.set, root=args.data, train=True, transform=train_transform)
  valid_data = build_dataset(args.set, root=args.data, train=False, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=args.num_workers)
  valid_queue = torch.utils.data.DataLoader(
      valid_data,
      batch_size=args.batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=args.num_workers)
  return train_queue, valid_queue


def train(train_queue, model, criterion, optimizer, args):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    x_adv = pgd_attack_train(
        model,
        input,
        target,
        eps=args.adv_eps,
        alpha=args.adv_alpha,
        steps=args.adv_steps,
        random_start=not args.no_random_start)

    model.train()
    optimizer.zero_grad()
    logits, logits_aux = logits_only(model(x_adv))
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight * loss_aux

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train_adv %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer_clean(valid_queue, model, criterion, args):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda(non_blocking=True)
      logits, _ = logits_only(model(input))
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid_clean %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer_pgd(valid_queue, model, criterion, args, steps):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)
    x_adv = pgd_attack_eval(
        model,
        input,
        target,
        eps=args.adv_eps,
        alpha=args.adv_alpha,
        steps=steps,
        batch_id=step,
        seed=args.seed)

    with torch.no_grad():
      logits, _ = logits_only(model(x_adv))
      loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid_pgd%d %03d %e %f %f', steps, step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def save_checkpoint(model, optimizer, scheduler, epoch, best_adv_acc, args):
  state = {
      'epoch': epoch,
      'state_dict': model.model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler.state_dict(),
      'best_adv_acc': best_adv_acc,
      'args': vars(args),
  }
  utils.save_checkpoint(state, False, args.save)


def main():
  args = parser.parse_args()
  if args.genotype_path is None and args.arch is None:
    parser.error('one of --genotype_path or --arch is required')
  if args.pgd_eval_freq is None:
    args.pgd_eval_freq = args.eval_freq

  args.save = 'adv-eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  set_seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled = True
  logging.info('gpu device = %d' % args.gpu)
  logging.info('args = %s', args)
  num_classes = 100 if args.set == 'cifar100' else 10
  cifar_mean, cifar_std = utils.get_cifar_mean_std(
      args.set,
      mean=args.cifar_mean,
      std=args.cifar_std)

  if args.genotype_path is not None:
    genotype = load_genotype(args.genotype_path)
  else:
    genotype = eval("genotypes.%s" % args.arch)
  logging.info('genotype = %s', genotype)
  with open(os.path.join(args.save, 'genotype.txt'), 'w') as f:
    f.write(str(genotype) + '\n')

  base_model = Network(args.init_channels, num_classes, args.layers, args.auxiliary, genotype)
  base_model.drop_path_prob = 0
  base_model = base_model.cuda()
  model = NormalizeWrapper(base_model, mean=cifar_mean, std=cifar_std).cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(base_model))

  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(
      optimizer,
      milestones=args.lr_decay_epochs,
      gamma=args.lr_decay_gamma)

  start_epoch = 0
  best_adv_acc = 0.0
  if args.resume:
    ckpt = torch.load(args.resume, map_location='cuda:%d' % args.gpu)
    base_model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best_adv_acc = ckpt.get('best_adv_acc', 0.0)
    logging.info('resumed from %s at epoch %d', args.resume, start_epoch)

  train_queue, valid_queue = build_loaders(args)

  for epoch in range(start_epoch, args.epochs):
    logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])
    base_model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer, args)
    logging.info('train_adv_acc %f', train_acc)

    should_clean_eval = args.clean_eval_freq > 0 and (
        ((epoch + 1) % args.clean_eval_freq == 0) or (epoch + 1 == args.epochs))
    should_pgd_eval = args.pgd_eval_freq > 0 and (
        ((epoch + 1) % args.pgd_eval_freq == 0) or (epoch + 1 == args.epochs))

    if should_clean_eval:
      valid_acc, valid_obj = infer_clean(valid_queue, model, criterion, args)
      logging.info('valid_clean_acc %f', valid_acc)

    if should_pgd_eval:
      adv_acc, adv_obj = infer_pgd(valid_queue, model, criterion, args, args.eval_pgd_steps)
      logging.info('valid_pgd%d_acc %f', args.eval_pgd_steps, adv_acc)

      if args.eval_pgd100_final and epoch + 1 == args.epochs:
        pgd100_acc, pgd100_obj = infer_pgd(valid_queue, model, criterion, args, 100)
        logging.info('valid_pgd100_acc %f', pgd100_acc)

      if adv_acc > best_adv_acc:
        best_adv_acc = adv_acc
        utils.save(base_model, os.path.join(args.save, 'best_adv_weights.pt'))

    utils.save(base_model, os.path.join(args.save, 'weights.pt'))
    save_checkpoint(model, optimizer, scheduler, epoch, best_adv_acc, args)
    scheduler.step()


if __name__ == '__main__':
  main()

import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

#FSP
from fsp_search import FSPHookConfig, run_fsp_guided_arch_update, should_run_fsp_hook

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'], help='dataset to use')
parser.add_argument('--cifar_mean', type=float, nargs=3, default=None, help='optional RGB normalization mean')
parser.add_argument('--cifar_std', type=float, nargs=3, default=None, help='optional RGB normalization std')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

#FSP
parser.add_argument('--fsp_hook_enable', action='store_true', default=False)
parser.add_argument('--fsp_start_epoch', type=int, default=16)
parser.add_argument('--fsp_interval_epochs', type=int, default=5)
parser.add_argument('--fsp_num_candidates', type=int, default=10)
parser.add_argument('--fsp_subset_num_folds', type=int, default=5)
parser.add_argument('--fsp_adv_ft_epochs', type=int, default=5)
parser.add_argument('--fsp_lambda_alpha', type=float, default=0.02)
parser.add_argument('--fsp_lambda_beta', type=float, default=0.01)
parser.add_argument('--fsp_eval_steps', type=int, default=7)


args = parser.parse_args()

'''
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
'''

CIFAR_CLASSES = 100 if args.set == 'cifar100' else 10
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  #
  args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  #FSP-cfg
  fsp_cfg = FSPHookConfig(
      start_epoch=args.fsp_start_epoch,
      interval_epochs=args.fsp_interval_epochs,
      num_candidates=args.fsp_num_candidates,
      subset_num_folds=args.fsp_subset_num_folds,
      adv_ft_epochs=args.fsp_adv_ft_epochs,
      lambda_alpha=args.fsp_lambda_alpha,
      lambda_beta=args.fsp_lambda_beta,
      eval_steps=args.fsp_eval_steps,
      dataset=args.set,
      mean=args.cifar_mean,
      std=args.cifar_std,
      data_path=args.data,
      seed=args.seed,
      batch_size=args.batch_size,
  )


  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  elif args.set=='svhn':
      train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    if epoch == args.epochs - 1:
        with open(os.path.join(args.save, 'genotype.txt'), 'w') as f:
            f.write(str(genotype))

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))
    print(F.softmax(model.betas_normal[2:5], dim=-1))
    #model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)

    #FSP
    if args.fsp_hook_enable:
      if epoch == args.epochs - 1:
        if should_run_fsp_hook(epoch, fsp_cfg):
          logging.info(
              "[FSP-HOOK] skip final epoch=%d to avoid final-only architecture bonus",
              epoch,
          )
        hook_summary = None
      else:
        hook_summary = run_fsp_guided_arch_update(
            model=model,
            epoch=epoch,
            device=torch.device('cuda'),
            cfg=fsp_cfg,
            logger=logging,
        )
      if hook_summary is not None:
          logging.info(
              "[FSP-HOOK] done: selected=%d/%d",
              hook_summary["num_selected"],
              hook_summary["num_candidates"],
          )

    # validation
    if args.epochs-epoch<=1:
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

    final_genotype = model.genotype()
    logging.info('Final Genotype: %s', final_genotype)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    #input_search, target_search = next(iter(valid_queue))
    try:
      input_search, target_search = next(valid_queue_iter)
    except:
      valid_queue_iter = iter(valid_queue)
      input_search, target_search = next(valid_queue_iter)
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)

    if epoch>=15:
      #check change for alpha/beta
      alpha_before = model.alphas_normal.clone()
      beta_before = model.betas_normal.clone()

      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

      #print
      with torch.no_grad():
          delta_alpha = (model.alphas_normal - alpha_before).abs().mean()
          delta_beta = (model.betas_normal - beta_before).abs().mean()

      if step % 50 == 0:
          logging.info(
              "ARCH STEP | Δalpha=%.6f Δbeta=%.6f",
              delta_alpha.item(),
              delta_beta.item()
          )

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():    
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda(non_blocking=True)
      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

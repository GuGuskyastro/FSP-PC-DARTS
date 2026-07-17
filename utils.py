import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

#10和100的mean/std
CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]
CIFAR100_MEAN = [0.50707459, 0.48654896, 0.44091788]
CIFAR100_STD = [0.26733422, 0.25643846, 0.27615058]
SVHN_MEAN = [0.43768218, 0.44376934, 0.47280428]
SVHN_STD = [0.19803010, 0.20101570, 0.19703591]


def get_cifar_mean_std(dataset='cifar10', mean=None, std=None):
  if mean is not None and std is not None:
    return mean, std

  dataset = dataset.lower()
  if dataset == 'cifar10':
    default_mean, default_std = CIFAR10_MEAN, CIFAR10_STD
  elif dataset == 'cifar100':
    default_mean, default_std = CIFAR100_MEAN, CIFAR100_STD
  elif dataset == 'svhn':
    default_mean, default_std = SVHN_MEAN, SVHN_STD
  else:
    raise ValueError('unsupported dataset: {}'.format(dataset))

  return mean or default_mean, std or default_std


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    #修改view到reshape
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  dataset = getattr(args, 'set', 'cifar10')
  CIFAR_MEAN, CIFAR_STD = get_cifar_mean_std(
    dataset,
    mean=getattr(args, 'cifar_mean', None),
    std=getattr(args, 'cifar_std', None),
  )

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

#修改了log创建逻辑
def create_exp_dir(path, scripts_to_save=None):
  os.makedirs(path, exist_ok=True)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import itertools
import random

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)
    self.k = 4

    self.selected_op = None

    for primitive in PRIMITIVES:
      op = OPS[primitive](C //self.k, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C //self.k, affine=False))
      self._ops.append(op)


  def forward(self, x, weights):
    #channel proportion k=4  
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//self.k, :, :]
    xtemp2 = x[ : ,  dim_2//self.k:, :, :]

    # constructing discrete network for evaluating FSP
    if self.selected_op is not None:
        temp1 = self._ops[self.selected_op](xtemp)
    else:
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))

    #reduction cell needs pooling before concat
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
    ans = channel_shuffle(ans,self.k)
    #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
    #except channe shuffle, channel shift also works
    return ans


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction
    #
    self.edge_mask = None

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights,weights2):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0

    for i in range(self._steps):
        s = 0
        for j, h in enumerate(states):
            edge_id = offset + j
            op = self._ops[edge_id]

            # use edge_mask to evaluate fixed architecture; otherwise follow the original PC-DARTS weighted search
            if self.edge_mask is not None:
                if not self.edge_mask[edge_id]:
                    continue

                s = s + op(h, None)
            else:
                s = s + weights2[edge_id] * op(h, weights[edge_id])

        offset += len(states)
        states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      s0, s1 = s1, cell(s0, s1, weights,weights2)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.betas_normal = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self.betas_reduce = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.betas_normal,
      self.betas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights,weights2):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
          W[j,:]=W[j,:]*W2[j]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        
        #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene
    n = 3
    start = 2
    weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    for i in range(self._steps-1):
      end = start + n
      tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
      tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype



  # method for constructing discrete models in FSP
  def _get_edge_weights_numpy(self, reduction: bool):
      if reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1).detach().cpu().numpy()
          betas = self.betas_reduce
      else:
          weights = F.softmax(self.alphas_normal, dim=-1).detach().cpu().numpy()
          betas = self.betas_normal

      n = 3
      start = 2
      weights2 = F.softmax(betas[0:2], dim=-1)
      for _ in range(self._steps - 1):
          end = start + n
          tw2 = F.softmax(betas[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2, tw2], dim=0)

      weights2 = weights2.detach().cpu().numpy()
      return weights, weights2

  def _enumerate_node_choices(self, W, per_edge_topops=2, per_node_topm=3):

      none_idx = PRIMITIVES.index('none')
      num_edges = W.shape[0]

      # keep top-k non-none ops for each edge
      edge_topops = {}
      for e in range(num_edges):
          cand = []
          for k in range(W.shape[1]):
              if k == none_idx:
                  continue
              cand.append((float(W[e, k]), PRIMITIVES[k], e))
          cand.sort(key=lambda x: x[0], reverse=True)
          edge_topops[e] = cand[:per_edge_topops]

      all_choices = []
      for e1, e2 in itertools.combinations(range(num_edges), 2):
          for c1 in edge_topops[e1]:
              for c2 in edge_topops[e2]:
                  score = c1[0] + c2[0]
                  gene_part = [(c1[1], c1[2]), (c2[1], c2[2])]
                  all_choices.append((score, gene_part))

      all_choices.sort(key=lambda x: x[0], reverse=True)
      return all_choices[:per_node_topm]

  def _parse_random(self, weights, weights2, num_samples=10, per_edge_topops=2, per_node_topm=3):

      per_node_candidates = []
      n = 2
      start = 0

      for i in range(self._steps):
          end = start + n
          W = weights[start:end].copy()
          W2 = weights2[start:end].copy()

          for j in range(n):
              W[j, :] = W[j, :] * W2[j]

          node_choices = self._enumerate_node_choices(
              W,
              per_edge_topops=per_edge_topops,
              per_node_topm=per_node_topm,
          )
          per_node_candidates.append(node_choices)

          start = end
          n += 1

      # candidate pool
      all_combos = []

      for combo in itertools.product(*per_node_candidates):

          score = sum(x[0] for x in combo)
          gene = []

          for _, g in combo:
              gene.extend(g)

          all_combos.append((score, gene))

      # random sample
      if len(all_combos) <= num_samples:
          sampled = all_combos
      else:
          sampled = random.sample(all_combos, num_samples)

      return sampled


  def genotypes_random(self, num_samples=10, per_edge_topops=2, per_node_topm=3):

      weights_n, weights2_n = self._get_edge_weights_numpy(reduction=False)
      weights_r, weights2_r = self._get_edge_weights_numpy(reduction=True)

      normal_samples = self._parse_random(
          weights_n, weights2_n,
          num_samples=num_samples,
          per_edge_topops=per_edge_topops,
          per_node_topm=per_node_topm,
      )

      reduce_samples = self._parse_random(
          weights_r, weights2_r,
          num_samples=num_samples,
          per_edge_topops=per_edge_topops,
          per_node_topm=per_node_topm,
      )

      concat = range(2 + self._steps - self._multiplier, self._steps + 2)

      combos = []

      for sn, gn in normal_samples:
          for sr, gr in reduce_samples:
              g = Genotype(
                  normal=gn, normal_concat=concat,
                  reduce=gr, reduce_concat=concat,
              )

              combos.append((sn + sr, g))

      if len(combos) > num_samples:
          combos = random.sample(combos, num_samples)

      return combos
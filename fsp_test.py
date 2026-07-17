import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from genotypes import PRIMITIVES
from model_search import Network
import argparse
from genotypes import Genotype
import random
import csv

#same in utils
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD  = [0.24703233, 0.24348505, 0.26158768]

#use seed to make results reproducible
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_genotype(path):
    with open(path, 'r') as f:
        genotype_str = f.read().strip()
    return eval(genotype_str, {"Genotype": Genotype})


#Normalize and make sure PGD in right scale
class NormalizeWrapper(nn.Module):
    def __init__(self, model, mean=None, std=None):
        super().__init__()
        self.model = model
        if mean is None:
            mean = CIFAR_MEAN
        if std is None:
            std = CIFAR_STD
        self.register_buffer(
            'mean', torch.tensor(mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor(std).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


#PGD Attacker same in RobNet, but PGD-7 too strong here set 1
class AttackPGD(nn.Module):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=1,base_seed=42):
        super().__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.base_seed = base_seed

    def forward(self, images, labels, batch_id):
        torch.manual_seed(self.base_seed + batch_id)
        torch.cuda.manual_seed_all(self.base_seed + batch_id)

        x = images.detach().clone()

        # random start
        x += torch.empty_like(x).uniform_(-self.eps, self.eps)
        x = torch.clamp(x, 0, 1)

        for _ in range(self.steps):
            x.requires_grad_()
            logits = self.model(x)
            loss = F.cross_entropy(logits, labels)

            grad = torch.autograd.grad(loss, x)[0]

            x = x.detach() + self.alpha * grad.sign()
            x = torch.min(torch.max(x, images - self.eps), images + self.eps)
            x = torch.clamp(x, 0, 1)

        return x.detach()



#fsp computation
def compute_fsp_matrix(feature_in, feature_out):
    B, C_in, H, W = feature_in.shape
    _, C_out, H2, W2 = feature_out.shape

    if (H, W) != (H2, W2):
        feature_out = F.interpolate(
            feature_out, size=(H, W),
            mode='bilinear', align_corners=False
        )

    f_in = feature_in.view(B, C_in, -1)
    f_out = feature_out.view(B, C_out, -1)

    fsp = torch.bmm(f_in, f_out.transpose(1, 2)) / (H * W)
    return fsp


def update_fsp_stats(fsp_stats, idx, fsp_clean, fsp_adv, mask=None):

    diff = (fsp_clean - fsp_adv) ** 2  # [B, Cin, Cout]
    per_sample = diff.mean(dim=(1, 2))  # [B]

    if mask is not None:
        if mask.sum() == 0:
            return
        per_sample = per_sample[mask]

    fsp_stats[idx]["sum"] += per_sample.sum().item()
    fsp_stats[idx]["count"] += per_sample.numel()



#Set a hook for fsp computation
features_buffer = {}

def get_cell_hook(cell_index):
    def hook(module,input_tuple, output):
        # s1 as input
        s1 = input_tuple[1]
        features_buffer[cell_index] = {
            'in': s1.detach(),
            'out': output.detach()
        }
    return hook


#Select discrete subnet with the genotype
def apply_genotype_mask(model, genotype):
    name_to_idx = {name: i for i, name in enumerate(PRIMITIVES)}

    for cell in model.cells:
        if cell.reduction:
            gene = genotype.reduce
        else:
            gene = genotype.normal

        k = sum(1 for i in range(cell._steps) for _ in range(2+i))
        cell.edge_mask = [False] * k

        offset = 0
        gene_ptr = 0

        for e in range(k):
            cell._ops[e].selected_op = None

        for i in range(cell._steps):
            edges = gene[gene_ptr:gene_ptr+2]
            gene_ptr += 2

            for op_name, src in edges:
                edge_id = offset + src
                cell.edge_mask[edge_id] = True

                op_idx = name_to_idx[op_name]
                cell._ops[edge_id].selected_op = op_idx

            offset += (i+2)


#finetune to each subnet
def freeze_arch_params(model):
    net = model.model if isinstance(model, NormalizeWrapper) else model
    if hasattr(net, "arch_parameters"):
        for p in net.arch_parameters():
            p.requires_grad_(False)


def finetune_subnet(model, loader, device, epochs, lr, weight_decay,
                   max_steps=None, print_freq=50):
    freeze_arch_params(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    step = 0
    for ep in range(epochs):
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if (step % print_freq) == 0:
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    acc = (pred == y).float().mean().item() * 100
                print(f"[FT] epoch {ep+1}/{epochs} step {step} loss {loss.item():.4f} acc {acc:.2f}%")

            step += 1
            if max_steps is not None and step >= max_steps:
                model.eval()
                return

    model.eval()

#update batch normalization
def recalibrate_bn(model, loader, device, num_batches):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None

    model.train()

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches: break
            model(x.to(device))

    model.eval()



#main for testing fsp in different attack strength for subnet
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genotype_path', type=str, required=True)
    parser.add_argument('--supernet_ckpt', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_channels = 16
    layers = 8
    num_classes = 10
    batch_size = 256
    data_path = './data'

    set_seed(42)

    #load supernet
    criterion = nn.CrossEntropyLoss()
    model = Network(init_channels, num_classes, layers, criterion).to(device)
    ckpt = torch.load(args.supernet_ckpt, map_location=device)

    model.load_state_dict(ckpt, strict=True)
    model.eval()

    genotype = load_genotype(args.genotype_path)

    apply_genotype_mask(model, genotype)

    wrapped_model = NormalizeWrapper(model).to(device)

    #set hooks
    target_cells = list(range(layers))
    hooks = []
    for idx in target_cells:
        hooks.append(model.cells[idx].register_forward_hook(get_cell_hook(idx)))

    set_seed(42)

    #CIFAR-10 test loader
    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_data = dset.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=valid_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    train_data_bn = dset.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=valid_transform
    )

    train_loader_bn = torch.utils.data.DataLoader(
        train_data_bn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    #fintune the subnet
    set_seed(42)
    finetune_subnet(
        wrapped_model,
        train_loader_bn,
        device,
        epochs=1,
        lr=0.01,
        weight_decay=3e-4,
        print_freq=50
    )

    set_seed(42)

    #update the batch normalization
    recalibrate_bn(wrapped_model, train_loader_bn, device, num_batches=200)


    #test fsp under different attack strength
    eps_list = [0.5/255, 1/255, 2/255, 4/255, 8/255]

    out_dir = "fsp_results"
    os.makedirs(out_dir, exist_ok=True)

    run_id = os.path.basename(os.path.dirname(args.genotype_path))
    csv_path = os.path.join(out_dir, f"{run_id}_fsp.csv")


    for eps in eps_list:

        attacker = AttackPGD(wrapped_model, eps=eps, alpha=eps/4, steps=1, base_seed=42)

        fsp_stats = {idx: {"sum": 0.0, "count": 0} for idx in target_cells}
        fsp_stats_clean_correct = {idx: {"sum": 0.0, "count": 0} for idx in target_cells}

        clean_correct = 0
        adv_correct = 0
        total = 0


        #evaluation loop
        for batch_id, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            raw_images = images

            features_buffer.clear()
            with torch.no_grad():
                logits = wrapped_model(raw_images)
            clean_feats = dict(features_buffer)

            pred = logits.argmax(dim=1)
            clean_correct += pred.eq(labels).sum().item()
            total += labels.size(0)


            adv_images = attacker(raw_images, labels, batch_id)

            features_buffer.clear()
            with torch.no_grad():
                adv_logits = wrapped_model(adv_images)
            adv_feats = features_buffer

            adv_pred = adv_logits.argmax(dim=1)
            adv_correct += adv_pred.eq(labels).sum().item()

            mask_clean_correct = pred.eq(labels)  # [B]

            for idx in target_cells:
                fsp_clean = compute_fsp_matrix(clean_feats[idx]['in'], clean_feats[idx]['out'])
                fsp_adv = compute_fsp_matrix(adv_feats[idx]['in'], adv_feats[idx]['out'])

                #all input
                update_fsp_stats(fsp_stats, idx, fsp_clean, fsp_adv, mask=None)

                #clean-correct input
                update_fsp_stats(fsp_stats_clean_correct, idx, fsp_clean, fsp_adv, mask_clean_correct)

        #report
        result_row = {}

        result_row["run_id"] = run_id

        #accuracy and robustness
        clean_acc = 100. * clean_correct / total
        adv_acc = 100. * adv_correct / total
        gap = clean_acc - adv_acc

        result_row["eps in 255"] = eps * 255
        result_row["clean_acc"] = clean_acc
        result_row["adv_acc"] = adv_acc
        result_row["robust_gap"] = gap

        #per-layer FSP
        cell_mean_fsp = {}
        cell_mean_fsp_clean_correct = {}

        for idx in target_cells:
            mean_fsp = fsp_stats[idx]["sum"] / max(fsp_stats[idx]["count"], 1)
            result_row[f"fsp_cell_{idx}"] = mean_fsp
            cell_mean_fsp[idx] = mean_fsp

            mean_fsp_cc = (fsp_stats_clean_correct[idx]["sum"] /max(fsp_stats_clean_correct[idx]["count"], 1))
            result_row[f"fsp_cell_{idx}_clean_correct"] = mean_fsp_cc
            cell_mean_fsp_clean_correct[idx] = mean_fsp_cc

        #average FSP
        result_row["fsp_mean_all"] = float(np.mean(list(cell_mean_fsp.values())))

        last4 = list(range(layers - 4, layers))
        result_row["fsp_mean_last4"] = float(np.mean([cell_mean_fsp[i] for i in last4]))

        result_row["fsp_mean_all_clean_correct"] = float(
            np.mean(list(cell_mean_fsp_clean_correct.values()))
        )

        result_row["fsp_mean_last4_clean_correct"] = float(
            np.mean([cell_mean_fsp_clean_correct[i] for i in last4])
        )

        print(result_row)

        #to csv
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(result_row)


    #cleanup
    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()

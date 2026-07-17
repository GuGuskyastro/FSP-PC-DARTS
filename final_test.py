import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils
from model import NetworkCIFAR as Network
from train_adv import NormalizeWrapper, build_dataset, load_genotype, set_seed
from fsp_test import (
    compute_fsp_matrix,
    features_buffer,
    get_cell_hook,
    update_fsp_stats,
)

def logits_only(output):
    if isinstance(output, tuple):
        return output[0]
    return output


@torch.enable_grad()
def pgd_attack(model, images, labels, eps, alpha, steps, batch_id, seed):
    was_training = model.training
    model.eval()

    torch.manual_seed(seed + batch_id)
    torch.cuda.manual_seed_all(seed + batch_id)

    x_adv = images.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_()
        logits = logits_only(model(x_adv))
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


def load_weights(model, weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=True)


def build_test_loader(dataset, data_path, batch_size, num_workers):
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_data = build_dataset(
        dataset,
        root=data_path,
        train=False,
        transform=valid_transform,
    )
    return torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def evaluate(model, test_loader, device, args):
    target_cells = list(range(args.layers))
    hooks = [model.model.cells[idx].register_forward_hook(get_cell_hook(idx)) for idx in target_cells]

    fsp_stats = {idx: {'sum': 0.0, 'count': 0} for idx in target_cells}
    clean_correct = 0
    adv_correct = 0
    total = 0

    model.eval()

    for batch_id, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        features_buffer.clear()
        with torch.no_grad():
            clean_logits = logits_only(model(images))
        clean_feats = dict(features_buffer)

        clean_pred = clean_logits.argmax(dim=1)
        clean_correct += clean_pred.eq(labels).sum().item()
        total += labels.size(0)

        adv_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.pgd_steps,
            batch_id=batch_id,
            seed=args.seed,
        )

        features_buffer.clear()
        with torch.no_grad():
            adv_logits = logits_only(model(adv_images))
        adv_feats = dict(features_buffer)

        adv_pred = adv_logits.argmax(dim=1)
        adv_correct += adv_pred.eq(labels).sum().item()

        for idx in target_cells:
            fsp_clean = compute_fsp_matrix(clean_feats[idx]['in'], clean_feats[idx]['out'])
            fsp_adv = compute_fsp_matrix(adv_feats[idx]['in'], adv_feats[idx]['out'])
            update_fsp_stats(fsp_stats, idx, fsp_clean, fsp_adv)

        if batch_id % args.report_freq == 0:
            clean_acc = 100.0 * clean_correct / max(total, 1)
            adv_acc = 100.0 * adv_correct / max(total, 1)
            print(
                f"[FINAL-TEST] batch {batch_id} "
                f"clean_acc {clean_acc:.4f} pgd{args.pgd_steps}_acc {adv_acc:.4f}"
            )

    for hook in hooks:
        hook.remove()

    cell_mean_fsp = {}
    result_row = {
        'run_id': args.run_id,
        'dataset': args.set,
        'genotype_path': args.genotype_path,
        'weights_path': args.weights_path,
        'clean_acc': 100.0 * clean_correct / max(total, 1),
        f'pgd{args.pgd_steps}_acc': 100.0 * adv_correct / max(total, 1),
        'eval_eps_in_255': args.eps * 255,
        'eval_alpha_in_255': args.alpha * 255,
        'layers': args.layers,
    }

    for idx in target_cells:
        mean_fsp = fsp_stats[idx]['sum'] / max(fsp_stats[idx]['count'], 1)
        result_row[f'fsp_cell_{idx}'] = mean_fsp
        cell_mean_fsp[idx] = mean_fsp

    last10 = list(range(args.layers - 10, args.layers))
    result_row['fsp_mean_all'] = float(np.mean(list(cell_mean_fsp.values())))
    result_row['fsp_mean_last10'] = float(np.mean([cell_mean_fsp[idx] for idx in last10]))

    return result_row


def main():
    parser = argparse.ArgumentParser("final_test")
    parser.add_argument('--genotype_path', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--set', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'], help='dataset to use')
    parser.add_argument('--cifar_mean', type=float, nargs=3, default=None, help='optional RGB normalization mean')
    parser.add_argument('--cifar_std', type=float, nargs=3, default=None, help='optional RGB normalization std')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='final_test_results')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--init_channels', type=int, default=36)
    parser.add_argument('--layers', type=int, default=20)
    parser.add_argument('--auxiliary', action='store_true', default=False)
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--alpha', type=float, default=2 / 255)
    parser.add_argument('--pgd_steps', type=int, default=20)
    parser.add_argument('--report_freq', type=int, default=10)
    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = os.path.basename(os.path.dirname(args.weights_path))

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    genotype = load_genotype(args.genotype_path)
    num_classes = 100 if args.set == 'cifar100' else 10
    cifar_mean, cifar_std = utils.get_cifar_mean_std(
        args.set,
        mean=args.cifar_mean,
        std=args.cifar_std,
    )

    base_model = Network(args.init_channels, num_classes, args.layers, args.auxiliary, genotype)
    base_model.drop_path_prob = 0
    load_weights(base_model, args.weights_path, device)
    base_model = base_model.to(device)
    model = NormalizeWrapper(base_model, mean=cifar_mean, std=cifar_std).to(device)

    test_loader = build_test_loader(args.set, args.data_path, args.batch_size, args.num_workers)
    result_row = evaluate(model, test_loader, device, args)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.run_id}_final_test.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

    print(result_row)
    print(f"Saved CSV: {csv_path}")


if __name__ == '__main__':
    main()

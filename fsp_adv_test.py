import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms

import utils
from model_search import Network
from fsp_test import (
    set_seed,
    load_genotype,
    NormalizeWrapper,
    AttackPGD,
    compute_fsp_matrix,
    update_fsp_stats,
    features_buffer,
    get_cell_hook,
    apply_genotype_mask,
    freeze_arch_params,
    recalibrate_bn,
    finetune_subnet,
)


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
            transform=transform,
        )
    return DatasetClass(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )


@torch.enable_grad()
def pgd_attack_train(model, images, labels, eps=8 / 255, alpha=2 / 255, steps=7, random_start=True):
    was_training = model.training
    model.eval()  # keep BN/dropout fixed while generating adversarial examples

    x_adv = images.detach().clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_()
        logits = model(x_adv)
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


def adv_finetune_subnet(model, loader, device, epochs, lr, weight_decay,
                        eps=8 / 255, alpha=2 / 255, steps=7, max_steps=None, print_freq=50):
    freeze_arch_params(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    global_step = 0

    for ep in range(epochs):
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            #adv samples
            x_adv = pgd_attack_train(
                model=model,
                images=x,
                labels=y,
                eps=eps,
                alpha=alpha,
                steps=steps,
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits_adv = model(x_adv)
            loss = criterion(logits_adv, y)
            loss.backward()
            optimizer.step()

            if (global_step % print_freq) == 0:
                with torch.no_grad():
                    adv_pred = logits_adv.argmax(dim=1)
                    adv_acc = (adv_pred == y).float().mean().item() * 100.0

                    clean_logits = model(x)
                    clean_pred = clean_logits.argmax(dim=1)
                    clean_acc = (clean_pred == y).float().mean().item() * 100.0

                print(
                    f"[ADV-FT] epoch {ep + 1}/{epochs} step {global_step} "
                    f"loss {loss.item():.4f} clean_acc {clean_acc:.2f}% adv_acc {adv_acc:.2f}%"
                )

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                model.eval()
                return

    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genotype_path', type=str, required=True)
    parser.add_argument('--supernet_ckpt', type=str, required=True)
    parser.add_argument('--set', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'], help='dataset to use')
    parser.add_argument('--cifar_mean', type=float, nargs=3, default=None, help='optional RGB normalization mean')
    parser.add_argument('--cifar_std', type=float, nargs=3, default=None, help='optional RGB normalization std')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)

    #short clean warm-up finetuning before adversarial finetuning
    parser.add_argument('--clean_ft_epochs', type=int, default=1)
    parser.add_argument('--clean_ft_lr', type=float, default=0.01)
    parser.add_argument('--clean_ft_weight_decay', type=float, default=3e-4)
    parser.add_argument('--clean_ft_max_steps', type=int, default=None)
    parser.add_argument('--clean_ft_print_freq', type=int, default=50)

    #short adversarial finetuning
    parser.add_argument('--adv_ft_epochs', type=int, default=5)
    parser.add_argument('--adv_ft_lr', type=float, default=0.005)
    parser.add_argument('--adv_ft_weight_decay', type=float, default=3e-4)
    parser.add_argument('--adv_ft_eps', type=float, default=8 / 255)
    parser.add_argument('--adv_ft_alpha', type=float, default=2 / 255)
    parser.add_argument('--adv_ft_steps', type=int, default=7)
    parser.add_argument('--adv_ft_max_steps', type=int, default=None)
    parser.add_argument('--adv_ft_print_freq', type=int, default=50)

    #20% dataset
    parser.add_argument('--use_ft_subset', action='store_true')
    parser.add_argument('--ft_subset_fold_id', type=int, default=0)
    parser.add_argument('--ft_subset_num_folds', type=int, default=5)

    #BN recalibration
    parser.add_argument('--bn_num_batches', type=int, default=200)

    #evaluation attacks
    parser.add_argument('--eval_steps', type=int, default=7)
    parser.add_argument('--eval_eps_list', type=float, nargs='+', default=[0.5 / 255, 1 / 255, 2 / 255, 4 / 255, 8 / 255])
    parser.add_argument('--eval_alpha_mode', type=str, default='eps_div_4', choices=['eps_div_4', 'fixed_2_255'])

    #output
    parser.add_argument('--out_dir', type=str, default='fsp_results_ablation/bonus')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_channels = 16
    layers = 8
    num_classes = 100 if args.set == 'cifar100' else 10
    cifar_mean, cifar_std = utils.get_cifar_mean_std(
        args.set,
        mean=args.cifar_mean,
        std=args.cifar_std,
    )

    set_seed(args.seed)

    #load supernet and discretize to subnet from genotype
    criterion = nn.CrossEntropyLoss()
    model = Network(init_channels, num_classes, layers, criterion).to(device)
    ckpt = torch.load(args.supernet_ckpt, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    genotype = load_genotype(args.genotype_path)
    apply_genotype_mask(model, genotype)
    wrapped_model = NormalizeWrapper(model, mean=cifar_mean, std=cifar_std).to(device)

    #hooks for FSP collection
    target_cells = list(range(layers))
    hooks = []
    for idx in target_cells:
        hooks.append(model.cells[idx].register_forward_hook(get_cell_hook(idx)))

    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_data = build_dataset(
        args.set,
        root=args.data_path,
        train=False,
        transform=valid_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    #20% dataset
    if args.use_ft_subset:
        from fsp_search import FSPHookConfig, build_finetune_loader

        tmp_cfg = FSPHookConfig(
            subset_ratio=1.0 / args.ft_subset_num_folds,
            subset_fold_id=args.ft_subset_fold_id,
            subset_num_folds=args.ft_subset_num_folds,
            batch_size=args.batch_size,
            num_workers=4,
            dataset=args.set,
            mean=args.cifar_mean,
            std=args.cifar_std,
            data_path=args.data_path,
            seed=args.seed,
        )
        train_loader_ft = build_finetune_loader(tmp_cfg, fold_id=args.ft_subset_fold_id)

    else:
        train_data_ft = build_dataset(
            args.set,
            root=args.data_path,
            train=True,
            transform=valid_transform
        )
        train_loader_ft = torch.utils.data.DataLoader(
            train_data_ft,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

    #short clean warm-up finetune
    if args.clean_ft_epochs > 0:
        set_seed(args.seed)
        finetune_subnet(
            wrapped_model,
            train_loader_ft,
            device,
            epochs=args.clean_ft_epochs,
            lr=args.clean_ft_lr,
            weight_decay=args.clean_ft_weight_decay,
            max_steps=args.clean_ft_max_steps,
            print_freq=args.clean_ft_print_freq,
        )

        #BN recalibration after clean warm-up
        set_seed(args.seed)
        recalibrate_bn(wrapped_model, train_loader_ft, device, num_batches=args.bn_num_batches)

    #short adversarial finetune
    if args.adv_ft_epochs > 0:
        set_seed(args.seed)
        adv_finetune_subnet(
            wrapped_model,
            train_loader_ft,
            device,
            epochs=args.adv_ft_epochs,
            lr=args.adv_ft_lr,
            weight_decay=args.adv_ft_weight_decay,
            eps=args.adv_ft_eps,
            alpha=args.adv_ft_alpha,
            steps=args.adv_ft_steps,
            max_steps=args.adv_ft_max_steps,
            print_freq=args.adv_ft_print_freq,
        )

    #final BN recalibration after all finetuning
    set_seed(args.seed)
    recalibrate_bn(wrapped_model, train_loader_ft, device, num_batches=args.bn_num_batches)

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = os.path.basename(os.path.dirname(args.genotype_path)) or os.path.splitext(os.path.basename(args.genotype_path))[0]
    csv_path = os.path.join(args.out_dir, f"{run_id}_fsp_adv_ft.csv")


    for eps in args.eval_eps_list:
        if args.eval_alpha_mode == 'eps_div_4':
            alpha = eps / 4
        else:
            alpha = 2 / 255

        attacker = AttackPGD(
            wrapped_model,
            eps=eps,
            alpha=alpha,
            steps=args.eval_steps,
            base_seed=args.seed,
        )

        fsp_stats = {idx: {"sum": 0.0, "count": 0} for idx in target_cells}
        fsp_stats_clean_correct = {idx: {"sum": 0.0, "count": 0} for idx in target_cells}

        clean_correct = 0
        adv_correct = 0
        total = 0

        for batch_id, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
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
            adv_feats = dict(features_buffer)

            adv_pred = adv_logits.argmax(dim=1)
            adv_correct += adv_pred.eq(labels).sum().item()

            mask_clean_correct = pred.eq(labels)

            for idx in target_cells:
                fsp_clean = compute_fsp_matrix(clean_feats[idx]['in'], clean_feats[idx]['out'])
                fsp_adv = compute_fsp_matrix(adv_feats[idx]['in'], adv_feats[idx]['out'])

                update_fsp_stats(fsp_stats, idx, fsp_clean, fsp_adv, mask=None)
                update_fsp_stats(fsp_stats_clean_correct, idx, fsp_clean, fsp_adv, mask_clean_correct)

        result_row = {}
        result_row['run_id'] = run_id
        result_row['dataset'] = args.set
        result_row['eval_eps_in_255'] = eps * 255
        result_row['eval_alpha_in_255'] = alpha * 255

        clean_acc = 100.0 * clean_correct / total
        adv_acc = 100.0 * adv_correct / total
        gap = clean_acc - adv_acc

        result_row['clean_acc'] = clean_acc
        result_row['adv_acc'] = adv_acc
        result_row['robust_gap'] = gap

        cell_mean_fsp = {}
        cell_mean_fsp_clean_correct = {}
        for idx in target_cells:
            mean_fsp = fsp_stats[idx]['sum'] / max(fsp_stats[idx]['count'], 1)
            mean_fsp_cc = fsp_stats_clean_correct[idx]['sum'] / max(fsp_stats_clean_correct[idx]['count'], 1)

            result_row[f'fsp_cell_{idx}'] = mean_fsp
            result_row[f'fsp_cell_{idx}_clean_correct'] = mean_fsp_cc
            cell_mean_fsp[idx] = mean_fsp
            cell_mean_fsp_clean_correct[idx] = mean_fsp_cc

        result_row['fsp_mean_all'] = float(np.mean(list(cell_mean_fsp.values())))
        result_row['fsp_mean_all_clean_correct'] = float(np.mean(list(cell_mean_fsp_clean_correct.values())))

        last4 = list(range(layers - 4, layers))
        result_row['fsp_mean_last4'] = float(np.mean([cell_mean_fsp[i] for i in last4]))
        result_row['fsp_mean_last4_clean_correct'] = float(
            np.mean([cell_mean_fsp_clean_correct[i] for i in last4])
        )

        print(result_row)

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_row)

    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()

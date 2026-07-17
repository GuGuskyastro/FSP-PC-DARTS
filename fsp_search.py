import copy
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import utils
from genotypes import PRIMITIVES
from fsp_test import (
    NormalizeWrapper,
    AttackPGD,
    compute_fsp_matrix,
    update_fsp_stats,
    features_buffer,
    get_cell_hook,
    freeze_arch_params,
    recalibrate_bn,
    finetune_subnet,
)
from fsp_adv_test import adv_finetune_subnet


def get_cifar_dataset_class(dataset: str):
    dataset = dataset.lower()
    if dataset == "cifar10":
        return dset.CIFAR10
    if dataset == "cifar100":
        return dset.CIFAR100
    if dataset == "svhn":
        return dset.SVHN
    raise ValueError(f"unsupported dataset: {dataset}")


def build_dataset(dataset: str, root: str, train: bool, transform):
    DatasetClass = get_cifar_dataset_class(dataset)
    if dataset.lower() == "svhn":
        split = "train" if train else "test"
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


@dataclass
class FSPHookConfig:
    start_epoch: int = 16
    interval_epochs: int = 5

    num_candidates: int = 10
    per_edge_topops: int = 2
    per_node_topm: int = 3

    subset_ratio: float = 0.2
    subset_fold_id: int = 0
    subset_num_folds: int = 5

    batch_size: int = 256
    num_workers: int = 2

    clean_ft_epochs: int = 1
    clean_ft_lr: float = 0.01
    clean_ft_weight_decay: float = 3e-4
    clean_ft_max_steps: int = None
    clean_ft_print_freq: int = 100

    adv_ft_epochs: int = 5
    adv_ft_lr: float = 0.005
    adv_ft_weight_decay: float = 3e-4
    adv_ft_eps: float = 8 / 255
    adv_ft_alpha: float = 2 / 255
    adv_ft_steps: int = 7
    adv_ft_max_steps: int = None
    adv_ft_print_freq: int = 100

    eval_eps: float = 8 / 255
    eval_alpha: float = 2 / 255
    eval_steps: int = 7

    bn_num_batches: int = 50

    select_fsp_ratio: float = 0.5
    select_acc_ratio: float = 0.5

    lambda_alpha: float = 0.02
    lambda_beta: float = 0.01

    use_last_n_cells: int = 4
    dataset: str = "cifar10"
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    data_path: str = "./data"
    seed: int = 2


def should_run_fsp_hook(epoch: int, cfg: FSPHookConfig) -> bool:
    if epoch < cfg.start_epoch:
        return False
    return ((epoch - cfg.start_epoch) % cfg.interval_epochs) == 0


def build_stratified_fold_indices(dataset, num_folds=5, seed=0):
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        targets = np.array(dataset.labels)
    else:
        raise AttributeError("dataset must expose targets or labels for stratified folds")
    rng = np.random.RandomState(seed)
    per_class_indices = {}

    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        per_class_indices[c] = np.array_split(idx, num_folds)

    folds = []
    for f in range(num_folds):
        fold_idx = []
        for c in sorted(per_class_indices.keys()):
            fold_idx.extend(per_class_indices[c][f].tolist())
        rng.shuffle(fold_idx)
        folds.append(fold_idx)
    return folds


def build_finetune_loader(cfg: FSPHookConfig, fold_id: int):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = build_dataset(
        cfg.dataset,
        root=cfg.data_path,
        train=True,
        transform=transform,
    )

    folds = build_stratified_fold_indices(
        train_data,
        num_folds=cfg.subset_num_folds,
        seed=cfg.seed,
    )
    subset = Subset(train_data, folds[fold_id % cfg.subset_num_folds])
    loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return loader


def build_eval_loader(cfg: FSPHookConfig):
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = build_dataset(
        cfg.dataset,
        root=cfg.data_path,
        train=False,
        transform=transform,
    )
    return DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


def clear_discrete_mask(model):
    for cell in model.cells:
        cell.edge_mask = None
        for op in cell._ops:
            op.selected_op = None


def apply_genotype_mask_inplace(model, genotype):
    op_to_idx = {name: i for i, name in enumerate(PRIMITIVES)}

    def build_cell_mask(gene, steps):
        edge_mask = []
        selected_ops = []

        offset = 0
        states_len = 2
        gene_ptr = 0

        for _ in range(steps):
            chosen = gene[gene_ptr:gene_ptr + 2]
            chosen_edges = {edge_idx: op_name for op_name, edge_idx in chosen}

            for j in range(states_len):
                if j in chosen_edges:
                    edge_mask.append(True)
                    selected_ops.append(op_to_idx[chosen_edges[j]])
                else:
                    edge_mask.append(False)
                    selected_ops.append(None)

            gene_ptr += 2
            offset += states_len
            states_len += 1

        return edge_mask, selected_ops

    edge_mask_n, selected_ops_n = build_cell_mask(genotype.normal, model._steps)
    edge_mask_r, selected_ops_r = build_cell_mask(genotype.reduce, model._steps)

    for cell in model.cells:
        if cell.reduction:
            cell.edge_mask = edge_mask_r
            for op, sel in zip(cell._ops, selected_ops_r):
                op.selected_op = sel
        else:
            cell.edge_mask = edge_mask_n
            for op, sel in zip(cell._ops, selected_ops_n):
                op.selected_op = sel


def evaluate_fsp_and_adv_acc(model, wrapped_model, eval_loader, device, cfg: FSPHookConfig):
    target_cells = list(range(model._layers))
    hooks = [model.cells[idx].register_forward_hook(get_cell_hook(idx)) for idx in target_cells]

    attacker = AttackPGD(
        wrapped_model,
        eps=cfg.eval_eps,
        alpha=cfg.eval_alpha,
        steps=cfg.eval_steps,
        base_seed=cfg.seed,
    )

    fsp_stats = {idx: {"sum": 0.0, "count": 0} for idx in target_cells}
    adv_correct = 0
    total = 0

    for batch_id, (images, labels) in enumerate(eval_loader):
        images = images.to(device)
        labels = labels.to(device)

        features_buffer.clear()
        with torch.no_grad():
            logits_clean = wrapped_model(images)
        clean_feats = dict(features_buffer)

        adv_images = attacker(images, labels, batch_id)

        features_buffer.clear()
        with torch.no_grad():
            logits_adv = wrapped_model(adv_images)
        adv_feats = dict(features_buffer)

        adv_pred = logits_adv.argmax(dim=1)
        adv_correct += adv_pred.eq(labels).sum().item()
        total += labels.size(0)

        for idx in target_cells:
            fsp_clean = compute_fsp_matrix(clean_feats[idx]['in'], clean_feats[idx]['out'])
            fsp_adv = compute_fsp_matrix(adv_feats[idx]['in'], adv_feats[idx]['out'])

            update_fsp_stats(fsp_stats, idx, fsp_clean, fsp_adv, mask=None)

    for h in hooks:
        h.remove()

    fsp_by_cell = {
        idx: (fsp_stats[idx]["sum"] / max(fsp_stats[idx]["count"], 1))
        for idx in target_cells
    }

    last_cells = target_cells[-cfg.use_last_n_cells:]
    fsp_mean_last = float(np.mean([fsp_by_cell[idx] for idx in last_cells]))
    adv_acc = 100.0 * adv_correct / max(total, 1)

    return {
        "adv_acc": adv_acc,
        "fsp_by_cell": fsp_by_cell,
        "fsp_mean_last": fsp_mean_last,
    }


def evaluate_one_candidate(supernet_model, genotype, ft_loader, eval_loader, device, cfg: FSPHookConfig):
    eval_model = copy.deepcopy(supernet_model).to(device)
    eval_model.eval()

    clear_discrete_mask(eval_model)
    apply_genotype_mask_inplace(eval_model, genotype)

    mean, std = utils.get_cifar_mean_std(cfg.dataset, mean=cfg.mean, std=cfg.std)
    wrapped_model = NormalizeWrapper(eval_model, mean=mean, std=std).to(device)

    if cfg.clean_ft_epochs > 0:
        finetune_subnet(
            wrapped_model,
            ft_loader,
            device,
            epochs=cfg.clean_ft_epochs,
            lr=cfg.clean_ft_lr,
            weight_decay=cfg.clean_ft_weight_decay,
            max_steps=cfg.clean_ft_max_steps,
            print_freq=cfg.clean_ft_print_freq,
        )
        recalibrate_bn(wrapped_model, ft_loader, device, num_batches=cfg.bn_num_batches)

    if cfg.adv_ft_epochs > 0:
        adv_finetune_subnet(
            wrapped_model,
            ft_loader,
            device,
            epochs=cfg.adv_ft_epochs,
            lr=cfg.adv_ft_lr,
            weight_decay=cfg.adv_ft_weight_decay,
            eps=cfg.adv_ft_eps,
            alpha=cfg.adv_ft_alpha,
            steps=cfg.adv_ft_steps,
            max_steps=cfg.adv_ft_max_steps,
            print_freq=cfg.adv_ft_print_freq,
        )

    recalibrate_bn(wrapped_model, ft_loader, device, num_batches=cfg.bn_num_batches)
    result = evaluate_fsp_and_adv_acc(eval_model, wrapped_model, eval_loader, device, cfg)

    del wrapped_model
    del eval_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def select_good_candidates(results: List[Dict[str, Any]], cfg: FSPHookConfig):
    if len(results) == 0:
        return []

    results_sorted_fsp = sorted(results, key=lambda x: x["fsp_mean_last"])
    keep_fsp = max(1, math.ceil(len(results_sorted_fsp) * cfg.select_fsp_ratio))
    stage1 = results_sorted_fsp[:keep_fsp]

    stage1_sorted_acc = sorted(stage1, key=lambda x: x["adv_acc"], reverse=True)
    keep_acc = max(1, math.ceil(len(stage1_sorted_acc) * cfg.select_acc_ratio))
    selected = stage1_sorted_acc[:keep_acc]
    return selected


def build_bonus_maps(model, selected_results):

    bonus_alpha_normal = torch.zeros_like(model.alphas_normal.data)
    bonus_alpha_reduce = torch.zeros_like(model.alphas_reduce.data)
    bonus_beta_normal = torch.zeros_like(model.betas_normal.data)
    bonus_beta_reduce = torch.zeros_like(model.betas_reduce.data)

    op_to_idx = {name: i for i, name in enumerate(PRIMITIVES)}
    num_selected = max(len(selected_results), 1)

    def accumulate_gene(gene, alpha_bonus, beta_bonus):
        offset = 0
        states_len = 2
        ptr = 0

        for _ in range(model._steps):
            chosen = gene[ptr:ptr + 2]
            for op_name, edge_idx in chosen:
                global_edge = offset + edge_idx
                alpha_bonus[global_edge, op_to_idx[op_name]] += 1.0 / num_selected
                beta_bonus[global_edge] += 1.0 / num_selected
            ptr += 2
            offset += states_len
            states_len += 1

    for item in selected_results:
        genotype = item["genotype"]
        accumulate_gene(genotype.normal, bonus_alpha_normal, bonus_beta_normal)
        accumulate_gene(genotype.reduce, bonus_alpha_reduce, bonus_beta_reduce)

    return {
        "alpha_normal": bonus_alpha_normal,
        "alpha_reduce": bonus_alpha_reduce,
        "beta_normal": bonus_beta_normal,
        "beta_reduce": bonus_beta_reduce,
    }

#check bonus results
def _tensor_delta_stats(delta: torch.Tensor):
    abs_delta = delta.abs()
    nonzero = (abs_delta > 0).float()
    return {
        "mean_abs": abs_delta.mean().item(),
        "max_abs": abs_delta.max().item(),
        "nonzero_ratio": nonzero.mean().item(),
    }


def _argmax_flip_stats(before_logits: torch.Tensor, after_logits: torch.Tensor, dim: int = -1):
    before_idx = before_logits.argmax(dim=dim)
    after_idx = after_logits.argmax(dim=dim)
    flipped = (before_idx != after_idx)

    return {
        "num_flipped": flipped.sum().item(),
        "flip_ratio": flipped.float().mean().item(),
    }


#apply bonus and record changes
@torch.no_grad()
def apply_arch_bonus(model, bonus_maps, cfg: FSPHookConfig):
    alpha_normal_before = model.alphas_normal.data.clone()
    alpha_reduce_before = model.alphas_reduce.data.clone()
    beta_normal_before = model.betas_normal.data.clone()
    beta_reduce_before = model.betas_reduce.data.clone()

    delta_alpha_normal = cfg.lambda_alpha * bonus_maps["alpha_normal"]
    delta_alpha_reduce = cfg.lambda_alpha * bonus_maps["alpha_reduce"]
    delta_beta_normal = cfg.lambda_beta * bonus_maps["beta_normal"]
    delta_beta_reduce = cfg.lambda_beta * bonus_maps["beta_reduce"]

    model.alphas_normal.data.add_(delta_alpha_normal)
    model.alphas_reduce.data.add_(delta_alpha_reduce)
    model.betas_normal.data.add_(delta_beta_normal)
    model.betas_reduce.data.add_(delta_beta_reduce)

    alpha_normal_after = model.alphas_normal.data
    alpha_reduce_after = model.alphas_reduce.data
    beta_normal_after = model.betas_normal.data
    beta_reduce_after = model.betas_reduce.data

    stats = {
        "delta_alpha_normal": _tensor_delta_stats(delta_alpha_normal),
        "delta_alpha_reduce": _tensor_delta_stats(delta_alpha_reduce),
        "delta_beta_normal": _tensor_delta_stats(delta_beta_normal),
        "delta_beta_reduce": _tensor_delta_stats(delta_beta_reduce),

        "flip_alpha_normal": _argmax_flip_stats(alpha_normal_before, alpha_normal_after, dim=-1),
        "flip_alpha_reduce": _argmax_flip_stats(alpha_reduce_before, alpha_reduce_after, dim=-1),
        "flip_beta_normal": _argmax_flip_stats(beta_normal_before, beta_normal_after, dim=-1),
        "flip_beta_reduce": _argmax_flip_stats(beta_reduce_before, beta_reduce_after, dim=-1),
    }

    return stats


def run_fsp_guided_arch_update(model, epoch, device, cfg: FSPHookConfig, logger=None):
    if not should_run_fsp_hook(epoch, cfg):
        return None

    if logger is not None:
        logger.info("[FSP-HOOK] start epoch=%d", epoch)

    ft_loader = build_finetune_loader(cfg, fold_id=(epoch - cfg.start_epoch) // cfg.interval_epochs)
    eval_loader = build_eval_loader(cfg)

    #candidate pool
    candidates = model.genotypes_random(
        num_samples=cfg.num_candidates,
        per_edge_topops=cfg.per_edge_topops,
        per_node_topm=cfg.per_node_topm,
    )

    candidate_results = []
    for rank, (score, genotype) in enumerate(candidates):
        if logger is not None:
            logger.info("[FSP-HOOK] evaluating sampled candidate %d/%d score=%.6f", rank + 1, len(candidates), score)

        metrics = evaluate_one_candidate(
            supernet_model=model,
            genotype=genotype,
            ft_loader=ft_loader,
            eval_loader=eval_loader,
            device=device,
            cfg=cfg,
        )
        metrics["genotype"] = genotype
        metrics["parse_score"] = score
        candidate_results.append(metrics)

        if logger is not None:
            logger.info(
                "[FSP-HOOK] candidate %d fsp=%.6f adv_acc=%.4f",
                rank + 1,
                metrics["fsp_mean_last"],
                metrics["adv_acc"],
            )

    #half fsp/ half acc
    selected = select_good_candidates(candidate_results, cfg)

    if logger is not None:
        logger.info("[FSP-HOOK] selected %d / %d candidates", len(selected), len(candidate_results))

    #bonus
    bonus_maps = build_bonus_maps(model, selected)
    bonus_stats = apply_arch_bonus(model, bonus_maps, cfg)

    #print changes
    logger.info(
        "[FSP-BONUS] alpha_normal: mean=%.6f max=%.6f nz=%.4f flip=%d(%.4f) | "
        "alpha_reduce: mean=%.6f max=%.6f nz=%.4f flip=%d(%.4f)",
        bonus_stats["delta_alpha_normal"]["mean_abs"],
        bonus_stats["delta_alpha_normal"]["max_abs"],
        bonus_stats["delta_alpha_normal"]["nonzero_ratio"],
        int(bonus_stats["flip_alpha_normal"]["num_flipped"]),
        bonus_stats["flip_alpha_normal"]["flip_ratio"],
        bonus_stats["delta_alpha_reduce"]["mean_abs"],
        bonus_stats["delta_alpha_reduce"]["max_abs"],
        bonus_stats["delta_alpha_reduce"]["nonzero_ratio"],
        int(bonus_stats["flip_alpha_reduce"]["num_flipped"]),
        bonus_stats["flip_alpha_reduce"]["flip_ratio"],
    )

    logger.info(
        "[FSP-BONUS] beta_normal: mean=%.6f max=%.6f nz=%.4f flip=%d(%.4f) | "
        "beta_reduce: mean=%.6f max=%.6f nz=%.4f flip=%d(%.4f)",
        bonus_stats["delta_beta_normal"]["mean_abs"],
        bonus_stats["delta_beta_normal"]["max_abs"],
        bonus_stats["delta_beta_normal"]["nonzero_ratio"],
        int(bonus_stats["flip_beta_normal"]["num_flipped"]),
        bonus_stats["flip_beta_normal"]["flip_ratio"],
        bonus_stats["delta_beta_reduce"]["mean_abs"],
        bonus_stats["delta_beta_reduce"]["max_abs"],
        bonus_stats["delta_beta_reduce"]["nonzero_ratio"],
        int(bonus_stats["flip_beta_reduce"]["num_flipped"]),
        bonus_stats["flip_beta_reduce"]["flip_ratio"],
    )

    summary = {
        "num_candidates": len(candidate_results),
        "num_selected": len(selected),
        "candidate_results": candidate_results,
        "selected_results": selected,
        "bonus_stats": bonus_stats,
    }
    return summary

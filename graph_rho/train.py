"""Training entry point for Graph-RHO."""
from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from graph_rho.config import DATASET_CONFIG, DEVICE_CONFIG, GNN_CONFIG
from graph_rho.gnn_data_loader import (
    GNNFlexibleDataset,
    compute_normalizers,
    get_dataloader,
    load_data,
)
from graph_rho.hetero_gnn_model import HeteroGNNModel
from graph_rho.utils.device_utils import get_device
from graph_rho.utils.path_utils import get_data_dir, get_log_dir, get_model_dir


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "1"}:
        return True
    if lowered in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def train_epoch(model, train_loader, optimizer, device, pos_weight=1.0,
                use_critical_path=True, critical_weight=0.5, critical_pos_weight=None):
    model.train()
    total_loss = 0.0
    total_loss_fix = 0.0
    total_loss_critical = 0.0
    num_batches = 0

    pos_weight_fix = torch.tensor([pos_weight], device=device)
    if critical_pos_weight is None:
        critical_pos_weight = pos_weight
    pos_weight_critical = torch.tensor([critical_pos_weight], device=device)

    for batch in tqdm(train_loader, total=len(train_loader), desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        if use_critical_path and getattr(model, "use_critical_path_head", False):
            output_fix, output_critical = model(batch, return_critical=True)
            loss_fix = F.binary_cross_entropy_with_logits(
                output_fix,
                batch.task_label.reshape(-1, 1),
                pos_weight=pos_weight_fix,
            )
            critical_label = getattr(batch, "task_critical_label", None)
            if critical_label is not None and len(critical_label) > 0:
                loss_critical = F.binary_cross_entropy_with_logits(
                    output_critical,
                    critical_label.reshape(-1, 1),
                    pos_weight=pos_weight_critical,
                )
            else:
                loss_critical = torch.tensor(0.0, device=device)
            loss = loss_fix + critical_weight * loss_critical
            total_loss_fix += loss_fix.item()
            total_loss_critical += loss_critical.item()
        else:
            output_fix = model(batch, return_critical=False)
            loss = F.binary_cross_entropy_with_logits(
                output_fix,
                batch.task_label.reshape(-1, 1),
                pos_weight=pos_weight_fix,
            )
            total_loss_fix += loss.item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    denom = max(num_batches, 1)
    return {
        "total": total_loss / denom,
        "fix": total_loss_fix / denom,
        "critical": total_loss_critical / denom if use_critical_path else 0.0,
    }


def evaluate(model, data_loader, device, pos_weight=1.0, thresholds=(0.5, 0.6),
             use_critical_path=True, critical_weight=0.5, critical_pos_weight=None):
    model.eval()
    total_loss = 0.0
    total_loss_fix = 0.0
    total_loss_critical = 0.0
    num_batches = 0

    pos_weight_fix = torch.tensor([pos_weight], device=device)
    if critical_pos_weight is None:
        critical_pos_weight = pos_weight
    pos_weight_critical = torch.tensor([critical_pos_weight], device=device)

    results_fix = {th: {"correct": 0, "total": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0} for th in thresholds}
    results_critical = {th: {"correct": 0, "total": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0} for th in thresholds}

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
            batch = batch.to(device)
            if use_critical_path and getattr(model, "use_critical_path_head", False):
                output_fix, output_critical = model(batch, return_critical=True)
                loss_fix = F.binary_cross_entropy_with_logits(
                    output_fix,
                    batch.task_label.reshape(-1, 1),
                    pos_weight=pos_weight_fix,
                )
                critical_label = getattr(batch, "task_critical_label", None)
                if critical_label is not None and len(critical_label) > 0:
                    loss_critical = F.binary_cross_entropy_with_logits(
                        output_critical,
                        critical_label.reshape(-1, 1),
                        pos_weight=pos_weight_critical,
                    )
                    probs_critical = torch.sigmoid(output_critical).cpu().numpy().reshape(-1)
                    labels_critical = critical_label.cpu().numpy()
                    for th in thresholds:
                        preds = (probs_critical >= th).astype(int)
                        results_critical[th]["correct"] += (preds == labels_critical).sum()
                        results_critical[th]["total"] += len(labels_critical)
                        results_critical[th]["TP"] += ((preds == 1) & (labels_critical == 1)).sum()
                        results_critical[th]["TN"] += ((preds == 0) & (labels_critical == 0)).sum()
                        results_critical[th]["FP"] += ((preds == 1) & (labels_critical == 0)).sum()
                        results_critical[th]["FN"] += ((preds == 0) & (labels_critical == 1)).sum()
                else:
                    loss_critical = torch.tensor(0.0, device=device)
                loss = loss_fix + critical_weight * loss_critical
                total_loss_fix += loss_fix.item()
                total_loss_critical += loss_critical.item()
            else:
                output_fix = model(batch, return_critical=False)
                loss = F.binary_cross_entropy_with_logits(
                    output_fix,
                    batch.task_label.reshape(-1, 1),
                    pos_weight=pos_weight_fix,
                )
                total_loss_fix += loss.item()

            total_loss += loss.item()
            num_batches += 1
            probs_fix = torch.sigmoid(output_fix).cpu().numpy().reshape(-1)
            labels_fix = batch.task_label.cpu().numpy()
            for th in thresholds:
                preds = (probs_fix >= th).astype(int)
                results_fix[th]["correct"] += (preds == labels_fix).sum()
                results_fix[th]["total"] += len(labels_fix)
                results_fix[th]["TP"] += ((preds == 1) & (labels_fix == 1)).sum()
                results_fix[th]["TN"] += ((preds == 0) & (labels_fix == 0)).sum()
                results_fix[th]["FP"] += ((preds == 1) & (labels_fix == 0)).sum()
                results_fix[th]["FN"] += ((preds == 0) & (labels_fix == 1)).sum()

    denom = max(num_batches, 1)
    metrics = {
        "loss": total_loss / denom,
        "loss_fix": total_loss_fix / denom,
        "loss_critical": total_loss_critical / denom if use_critical_path else 0.0,
    }

    for th in thresholds:
        result = results_fix[th]
        accuracy = result["correct"] / max(result["total"], 1)
        precision = result["TP"] / max(result["TP"] + result["FP"], 1)
        recall = result["TP"] / max(result["TP"] + result["FN"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        metrics[f"accuracy_{th}"] = accuracy
        metrics[f"precision_{th}"] = precision
        metrics[f"recall_{th}"] = recall
        metrics[f"f1_{th}"] = f1
        metrics[f"TP_{th}"] = result["TP"]
        metrics[f"TN_{th}"] = result["TN"]
        metrics[f"FP_{th}"] = result["FP"]
        metrics[f"FN_{th}"] = result["FN"]

    if use_critical_path:
        for th in thresholds:
            result = results_critical[th]
            if result["total"] == 0:
                continue
            accuracy = result["correct"] / max(result["total"], 1)
            precision = result["TP"] / max(result["TP"] + result["FP"], 1)
            recall = result["TP"] / max(result["TP"] + result["FN"], 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            metrics[f"critical_accuracy_{th}"] = accuracy
            metrics[f"critical_precision_{th}"] = precision
            metrics[f"critical_recall_{th}"] = recall
            metrics[f"critical_f1_{th}"] = f1

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, save_dir, checkpoint_name, config=None, silent=False):
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config or GNN_CONFIG.copy(),
    }
    path = save_dir / f"{checkpoint_name}.pth"
    torch.save(checkpoint, path)
    if not silent:
        print(f"Saved checkpoint to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Train Graph-RHO on makespan data")
    parser.add_argument("--num_jobs", type=int, default=DATASET_CONFIG["num_jobs"])
    parser.add_argument("--num_machines", type=int, default=DATASET_CONFIG["num_machines"])
    parser.add_argument("--num_ops_per_job", type=int, default=DATASET_CONFIG["num_ops_per_job"])
    parser.add_argument("--instance_type", type=str, default=DATASET_CONFIG["instance_type"])
    parser.add_argument("--window", type=int, default=DATASET_CONFIG["window"])
    parser.add_argument("--step", type=int, default=DATASET_CONFIG["step"])
    parser.add_argument("--time_limit", type=int, default=DATASET_CONFIG["time_limit"])
    parser.add_argument("--stop_search_time", type=int, default=DATASET_CONFIG["stop_search_time"])
    parser.add_argument("--num_epochs", type=int, default=GNN_CONFIG["num_epochs"])
    parser.add_argument("--lr", type=float, default=GNN_CONFIG["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=GNN_CONFIG["batch_size"])
    parser.add_argument("--pos_weight", type=float, default=GNN_CONFIG["pos_weight"])
    parser.add_argument("--hidden_dim", type=int, default=GNN_CONFIG["hidden_dim"])
    parser.add_argument("--num_gnn_layers", type=int, default=GNN_CONFIG["num_gnn_layers"])
    parser.add_argument("--gnn_type", type=str, default=GNN_CONFIG["gnn_type"], choices=["gat", "sage", "gcn"])
    parser.add_argument("--num_heads", type=int, default=GNN_CONFIG["num_attention_heads"])
    parser.add_argument("--dropout", type=float, default=GNN_CONFIG["dropout"])
    parser.add_argument("--train_start", type=int, default=DATASET_CONFIG["train_start"])
    parser.add_argument("--train_end", type=int, default=DATASET_CONFIG["train_end"])
    parser.add_argument("--val_start", type=int, default=DATASET_CONFIG["val_start"])
    parser.add_argument("--val_end", type=int, default=DATASET_CONFIG["val_end"])
    parser.add_argument("--eval_every", type=int, default=GNN_CONFIG["eval_every"])
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--use_critical_path", type=str2bool, default=GNN_CONFIG["use_critical_path"])
    parser.add_argument("--critical_weight", type=float, default=GNN_CONFIG["critical_weight"])
    parser.add_argument("--critical_pos_weight", type=float, default=GNN_CONFIG["critical_pos_weight"])
    args = parser.parse_args()

    dataset_config = {
        "num_jobs": args.num_jobs,
        "num_machines": args.num_machines,
        "num_ops_per_job": args.num_ops_per_job,
        "instance_type": args.instance_type,
        "window": args.window,
        "step": args.step,
        "time_limit": args.time_limit,
        "stop_search_time": args.stop_search_time,
    }

    device = get_device(DEVICE_CONFIG["prefer"])
    print(f"Using device: {device}")

    data_dir = get_data_dir()
    model_root = get_model_dir()
    log_root = get_log_dir()

    if args.model_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        args.model_name = f"graph_rho_{args.gnn_type}_{timestamp}"

    model_dir = model_root / args.model_name
    log_dir = log_root / args.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    instance_name = (
        f"j{dataset_config['num_jobs']}-m{dataset_config['num_machines']}-"
        f"t{dataset_config['num_ops_per_job']}_{dataset_config['instance_type']}"
    )
    print("\n" + "=" * 60)
    print("Training Graph-RHO")
    print("=" * 60)
    print(f"Model name: {args.model_name}")
    print(f"Instance family: {instance_name}")
    print(f"Window/step: {dataset_config['window']}/{dataset_config['step']}")
    print(f"Time limit / stop-search: {dataset_config['time_limit']} / {dataset_config['stop_search_time']}")

    print("\nLoading training data...")
    train_data = load_data(data_dir, args.train_start, args.train_end, dataset_config)
    if not train_data:
        print("\nERROR: no training data found under Graph-RHO/data.")
        print("Prepare makespan training samples first via the patched L-RHO pipeline.")
        return

    print("Loading validation data...")
    val_data = load_data(data_dir, args.val_start, args.val_end, dataset_config)
    if not val_data:
        print("WARNING: no validation data found; using 10% of training data instead.")
        val_data = train_data[: max(1, len(train_data) // 10)]

    print("\nComputing normalization statistics...")
    x_tasks_norm, x_machines_norm = compute_normalizers(train_data)
    normalizer_dir = model_dir / "input_normalizer"
    normalizer_dir.mkdir(parents=True, exist_ok=True)
    x_tasks_norm.save(normalizer_dir / "normalizer_tasks.pkl")
    x_machines_norm.save(normalizer_dir / "normalizer_machines.pkl")

    train_dataset = GNNFlexibleDataset(train_data)
    val_dataset = GNNFlexibleDataset(val_data)
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = HeteroGNNModel(
        input_task_dim=GNN_CONFIG["input_task_dim"],
        input_machine_dim=GNN_CONFIG["input_machine_dim"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_gnn_layers,
        gnn_type=args.gnn_type,
        num_heads=args.num_heads,
        dropout=args.dropout,
        machine_task_edge_dim=GNN_CONFIG["machine_task_edge_dim"],
        task_precedence_edge_dim=GNN_CONFIG["task_precedence_edge_dim"],
        task_solution_edge_dim=GNN_CONFIG["task_solution_edge_dim"],
        use_global_aggr=GNN_CONFIG["use_global_aggr"],
        x_tasks_norm=x_tasks_norm,
        x_machines_norm=x_machines_norm,
        use_critical_path_head=args.use_critical_path,
    ).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=GNN_CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    writer = SummaryWriter(log_dir=str(log_dir))

    config_path = model_dir / "config.pkl"
    with open(config_path, "wb") as handle:
        pickle.dump(
            {
                "args": vars(args),
                "gnn_config": GNN_CONFIG,
                "dataset_config": dataset_config,
            },
            handle,
        )

    training_config = GNN_CONFIG.copy()
    training_config.update(
        {
            "hidden_dim": args.hidden_dim,
            "num_gnn_layers": args.num_gnn_layers,
            "gnn_type": args.gnn_type,
            "num_attention_heads": args.num_heads,
            "dropout": args.dropout,
            "use_critical_path": args.use_critical_path,
            "critical_weight": args.critical_weight,
        }
    )

    best_val_f1 = 0.0
    best_epoch = 0
    print("\n" + "=" * 60)
    print(f"Starting training for {args.num_epochs} epochs")
    print("=" * 60)
    for epoch in range(args.num_epochs):
        start = time.time()
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.pos_weight,
            use_critical_path=args.use_critical_path,
            critical_weight=args.critical_weight,
            critical_pos_weight=args.critical_pos_weight,
        )
        writer.add_scalar("train/loss", train_losses["total"], epoch)
        writer.add_scalar("train/loss_fix", train_losses["fix"], epoch)
        if args.use_critical_path:
            writer.add_scalar("train/loss_critical", train_losses["critical"], epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
            train_metrics = evaluate(
                model,
                train_loader,
                device,
                args.pos_weight,
                use_critical_path=args.use_critical_path,
                critical_weight=args.critical_weight,
                critical_pos_weight=args.critical_pos_weight,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args.pos_weight,
                use_critical_path=args.use_critical_path,
                critical_weight=args.critical_weight,
                critical_pos_weight=args.critical_pos_weight,
            )
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            val_f1 = val_metrics.get("f1_0.5", 0.0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, val_metrics, model_dir, f"{args.model_name}_best", config=training_config)
            save_checkpoint(model, optimizer, epoch, val_metrics, model_dir, f"{args.model_name}_last", config=training_config, silent=True)
            elapsed = time.time() - start
            print(f"\nEpoch [{epoch + 1}/{args.num_epochs}] Time: {elapsed:.1f}s")
            print(
                f"  Loss: Total={train_losses['total']:.4f}, Fix={train_losses['fix']:.4f}, "
                f"Critical={train_losses['critical']:.4f}"
            )
            print(
                f"  Fix Task    - Precision: {val_metrics.get('precision_0.5', 0):.4f}, "
                f"Recall: {val_metrics.get('recall_0.5', 0):.4f}, F1: {val_metrics.get('f1_0.5', 0):.4f}"
            )
            if args.use_critical_path and "critical_precision_0.5" in val_metrics:
                print(
                    f"  Critical Task - Precision: {val_metrics.get('critical_precision_0.5', 0):.4f}, "
                    f"Recall: {val_metrics.get('critical_recall_0.5', 0):.4f}, "
                    f"F1: {val_metrics.get('critical_f1_0.5', 0):.4f}"
                )
        else:
            elapsed = time.time() - start
            print(f"Epoch [{epoch + 1}/{args.num_epochs}] Loss: {train_losses['total']:.4f} | Time: {elapsed:.1f}s")
        scheduler.step()

    writer.close()
    print("\n" + "=" * 60)
    print("Training completed")
    print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch + 1}")
    print(f"Checkpoints saved to: {model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

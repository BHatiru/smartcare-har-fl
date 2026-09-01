"""
train_centralized.py — Centralized STGCN++ training for medical action recognition.

Supports both 2D (COCO-17) and 3D (NTU-25) skeleton data with cosine LR scheduling,
warmup, and proper experiment tracking.

Usage:
    # 2D centralized
    python train_centralized.py \
        --data-dir data/fed_medical_2d/centralized \
        --graph coco --num-classes 15 --epochs 80 --batch-size 128 --device cuda

    # 3D centralized
    python train_centralized.py \
        --data-dir data/fed_medical_3d/centralized \
        --graph ntu --num-classes 15 --epochs 80 --batch-size 128 --device cuda
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.stgcn import build_model, count_parameters
from prepare_medical_data import MEDICAL_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_centralized")


def load_data(data_dir: Path, batch_size: int):
    """Load train/test NPZ files into DataLoaders."""
    train_data = np.load(data_dir / 'train.npz')
    test_data = np.load(data_dir / 'test.npz')

    x_train = torch.from_numpy(train_data['x'].astype(np.float32))
    y_train = torch.from_numpy(train_data['y'].astype(np.int64))
    x_test = torch.from_numpy(test_data['x'].astype(np.float32))
    y_test = torch.from_numpy(test_data['y'].astype(np.int64))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    logger.info(f"Train: {x_train.shape} ({len(y_train)} samples)")
    logger.info(f"Test:  {x_test.shape} ({len(y_test)} samples)")
    return train_loader, test_loader


def get_cosine_lr(epoch, total_epochs, base_lr, warmup_epochs=5, min_lr=1e-5):
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    total_epochs, base_lr, warmup_epochs):
    """Train for one epoch with per-epoch cosine LR."""
    lr = get_cosine_lr(epoch, total_epochs, base_lr, warmup_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, correct / total, lr


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    all_preds, all_labels = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * yb.size(0)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(yb.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Centralized STGCN++ training')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--graph', choices=['coco', 'ntu'], default='coco')
    parser.add_argument('--num-classes', type=int, default=15)
    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--num-stages', type=int, default=6)
    parser.add_argument('--num-person', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience (0=disabled). Stop if no '
                             'improvement for this many epochs after warmup.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--tag', type=str, default=None,
                        help='Experiment tag for checkpoint naming')
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tag for file naming
    tag = args.tag or f"cent_{'3d' if args.graph == 'ntu' else '2d'}_{args.num_classes}cls"

    # Load data
    train_loader, test_loader = load_data(data_dir, args.batch_size)

    # Build model
    model = build_model(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        num_person=args.num_person,
        dropout=args.dropout,
        graph=args.graph,
    ).to(device)

    logger.info(f"Model: STGCN++ ({args.graph}) | params: {count_parameters(model):,}")
    logger.info(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, "
                f"lr={args.lr}, warmup={args.warmup_epochs}"
                + (f", patience={args.patience}" if args.patience > 0 else ""))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=args.weight_decay, nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history = []
    start_time = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc, lr = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.epochs, args.lr, args.warmup_epochs,
        )
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / f'{tag}_best.pt')
        else:
            epochs_without_improvement += 1

        record = {
            'epoch': epoch + 1,
            'lr': round(lr, 6),
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'test_loss': round(test_loss, 4),
            'test_acc': round(test_acc, 4),
            'best_acc': round(best_acc, 4),
            'time_s': round(elapsed, 1),
        }
        history.append(record)

        marker = ' ★' if is_best else ''
        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs}  |  "
            f"lr={lr:.5f}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}{marker}  |  "
            f"{elapsed:.1f}s"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), output_dir / f'{tag}_e{epoch+1}.pt')

        # Early stopping (only after warmup)
        if (args.patience > 0
                and (epoch + 1) > args.warmup_epochs
                and epochs_without_improvement >= args.patience):
            logger.info(f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {args.patience} epochs, "
                        f"best={best_acc:.4f} at epoch {best_epoch})")
            stopped_early = True
            break

    actual_epochs = epoch + 1
    total_time = time.time() - start_time

    # Final evaluation with per-class metrics
    _, final_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    # Per-class accuracy
    class_correct = Counter()
    class_total = Counter()
    for p, l in zip(preds, labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    # Confusion matrix (rows=true, cols=pred)
    num_cls = args.num_classes
    confusion = np.zeros((num_cls, num_cls), dtype=int)
    for p, l in zip(preds, labels):
        confusion[l][p] += 1

    # Label names
    label_names = MEDICAL_LABELS[:num_cls]

    # Save final model
    torch.save(model.state_dict(), output_dir / f'{tag}_final.pt')

    # Save confusion matrix as CSV for easy import
    import csv
    cm_path = output_dir / f'{tag}_confusion_matrix.csv'
    with open(cm_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + label_names)
        for i, row in enumerate(confusion):
            writer.writerow([label_names[i]] + row.tolist())

    # Save history
    result = {
        'tag': tag,
        'graph': args.graph,
        'num_classes': args.num_classes,
        'epochs': actual_epochs,
        'max_epochs': args.epochs,
        'patience': args.patience,
        'stopped_early': stopped_early,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'dropout': args.dropout,
        'base_channels': args.base_channels,
        'num_stages': args.num_stages,
        'num_person': args.num_person,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': final_acc,
        'total_time_s': round(total_time, 1),
        'params': count_parameters(model),
        'label_names': label_names,
        'per_class_acc': {
            int(c): {
                'name': label_names[c] if c < len(label_names) else f'class_{c}',
                'accuracy': round(class_correct[c] / max(class_total[c], 1), 4),
                'correct': class_correct[c],
                'total': class_total[c],
            }
            for c in sorted(class_total.keys())
        },
        'confusion_matrix': confusion.tolist(),
        'history': history,
    }
    with open(output_dir / f'{tag}_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\n{'═' * 60}")
    print(f"  Centralized Training Complete — {tag}")
    print(f"{'═' * 60}")
    print(f"  Best acc:  {best_acc:.4f} (epoch {best_epoch})")
    print(f"  Final acc: {final_acc:.4f}")
    print(f"  Time:      {total_time / 60:.1f} min")
    print(f"  Checkpoint: {output_dir / f'{tag}_best.pt'}")
    print(f"  Results:    {output_dir / f'{tag}_results.json'}")
    print(f"  Confusion:  {cm_path}")
    print(f"\n  Per-class accuracy:")
    for c in sorted(class_total.keys()):
        acc = class_correct[c] / max(class_total[c], 1)
        name = label_names[c] if c < len(label_names) else f'class_{c}'
        print(f"    [{c:2d}] {name:20s} {acc:.4f}  ({class_correct[c]}/{class_total[c]})")
    print(f"{'═' * 60}")


if __name__ == '__main__':
    main()

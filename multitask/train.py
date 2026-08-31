"""Multi-task training: classification (benign/malignant) + segmentation (lesion).

5-fold stratified cross-validation (same idea as the paper). Each fold trains a
fresh model from the ImageNet-pretrained backbone, validates on the held-out
fold, and reports per-fold best metrics + mean±std at the end.

Saving scheme:
  output/best_seg.pth              -- GLOBAL best segmentation model (by val Dice), updated when any fold beats it
  output/best_cls.pth              -- GLOBAL best classification model (by val ACC), updated when any fold beats it
  output/exp_{TS}/fold_{k}/ckpt_epochN.tar  -- resume checkpoint every --save-interval epochs
                                      (epoch + model + optimizer + scheduler);
                                      keeps the --max-ckpt most recent of THAT fold
  output/exp_{TS}/fold_{k}/train_log.csv    -- full per-epoch metric history for that fold
  output/exp_{TS}/cv_summary.csv    -- per-fold best metrics + mean/std
  runs/exp_{TS}/fold_{k}/          -- TensorBoard events (visualization only)

Usage:
  python multitask/train.py                              # 5-fold CV (defaults)
  python multitask/train.py --k-folds 10                 # 10-fold
  python multitask/train.py --fold 2                     # run only fold 2 (0-indexed)
  python multitask/train.py --fold 2 --resume output/exp_XXX/fold_2/ckpt_epoch30.tar --epochs 60
"""
import argparse
import copy
import csv
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from multitask.model import MultiTaskVMamba, VSSM_TINY
from multitask.dataset import BUSIDataset

PRETRAINED_PATH = os.path.join(_ROOT, "checkpoints", "pretrained", "vssm_tiny_0230_ckpt_epoch_262.pth")


# ----------------------------- losses & metrics -----------------------------

def dice_loss(logits, target, smooth=1.0):
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + smooth) / (union + smooth)
    return (1 - dice).mean()


def focal_loss(logits, targets, gamma=2.0):
    """Focal loss: down-weight well-classified examples (handles imbalance
    without pushing a class prior, unlike inverse-frequency weighting)."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def tversky_loss(logits, target, alpha=0.3, beta=0.7, smooth=1.0):
    """Tversky loss: alpha weights FP, beta weights FN. beta>alpha penalises
    false negatives (missed lesions) more, which helps small UCLM lesions."""
    pred = torch.sigmoid(logits)
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tversky).mean()


def fda(images, beta=0.15):
    """Fourier Domain Adaptation: swap the low-frequency amplitude spectrum
    between random pairs in the batch (keeps content/high-freq, swaps style/domain)."""
    B = images.size(0)
    if B < 2:
        return images
    fft = torch.fft.rfft2(images, norm="ortho")
    amp, pha = torch.abs(fft), torch.angle(fft)
    _, _, H, W = amp.shape
    low = torch.zeros_like(amp)
    low[..., : int(H * beta), : int(W * beta)] = 1.0
    perm = torch.randperm(B, device=images.device)
    new_amp = amp * (1 - low) + amp[perm] * low
    new_fft = new_amp * torch.exp(1j * pha)
    return torch.fft.irfft2(new_fft, s=(images.size(-2), images.size(-1)), norm="ortho")


def compute_metrics(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union_seg = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    union_iou = (pred + target).clamp(0, 1).sum(dim=(1, 2, 3))
    dice = ((2 * inter + 1e-6) / (union_seg + 1e-6)).mean().item()
    iou = ((inter + 1e-6) / (union_iou + 1e-6)).mean().item()
    return dice, iou


def evaluate(model, loader, device, criterion_cls, criterion_seg):
    """Return (cls_loss, seg_loss, acc, auc, dice, iou) averaged over the loader."""
    model.eval()
    total, correct = 0, 0
    tot_cls_loss = tot_seg_loss = 0.0
    tot_dice = tot_iou = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, masks, labels in loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            logits, mask_logits = model(images)
            loss_cls = criterion_cls(logits, labels)
            loss_seg = criterion_seg(mask_logits, masks) + dice_loss(mask_logits, masks)
            tot_cls_loss += loss_cls.item() * images.size(0)
            tot_seg_loss += loss_seg.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
            d, i = compute_metrics(mask_logits, masks)
            tot_dice += d * images.size(0)
            tot_iou += i * images.size(0)
            all_logits.append(logits)
            all_labels.append(labels)

    n = max(total, 1)
    # Binary ROC-AUC from softmax probability of class 1.
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    proba = torch.softmax(all_logits, dim=1)[:, 1].cpu().numpy()
    y = all_labels.cpu().numpy()
    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.0

    return (tot_cls_loss / n, tot_seg_loss / n, correct / n, auc, tot_dice / n, tot_iou / n)


def _ckpt_epoch(path):
    """Extract the epoch number from a ckpt_epochN.tar filename."""
    return int(os.path.basename(path).split("_epoch")[1].split(".tar")[0])


def _load_global_best(path):
    """Return the persisted global-best metric from a best_*.pth file (0.0 if absent)."""
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "metric" in ckpt:
            return ckpt["metric"]
    return 0.0


def _build_model(device, pretrained, use_lass=False, use_lesion_pool=False):
    pretrained = pretrained if os.path.exists(pretrained) else None
    if pretrained is None:
        print(f"[train] WARNING: pretrained weight not found — training from scratch")
    model = MultiTaskVMamba(encoder_config=VSSM_TINY, num_classes=2,
                            pretrained=pretrained, use_lass=use_lass,
                            use_lesion_pool=use_lesion_pool)
    return model.to(device)


# ----------------------------- main ----------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="dataset/Dataset_BUSI_with_GT")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-folds", type=int, default=5, help="number of CV folds")
    parser.add_argument("--fold", type=int, default=-1, help="run only this fold index (default: all)")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--tb-dir", type=str, default="runs")
    parser.add_argument("--save-interval", type=int, default=10, help="save a resume ckpt every N epochs")
    parser.add_argument("--max-ckpt", type=int, default=3, help="keep at most N most recent ckpts per fold")
    parser.add_argument("--pretrained", type=str, default=PRETRAINED_PATH)
    parser.add_argument("--resume", type=str, default="", help="path to a fold ckpt_*.tar to resume")
    parser.add_argument("--no-tb", action="store_true", help="disable TensorBoard logging")
    parser.add_argument("--smoke", action="store_true", help="run a few batches to verify the pipeline")
    parser.add_argument("--exclude-prefix", type=str, default="",
                        help="leave-one-out: drop samples whose image basename starts with this prefix")
    parser.add_argument("--class-weight", action="store_true",
                        help="weight CrossEntropyLoss by inverse class frequency")
    parser.add_argument("--focal", action="store_true", help="use focal loss for classification")
    parser.add_argument("--tversky", action="store_true", help="use Tversky loss for segmentation")
    parser.add_argument("--fda", action="store_true", help="apply Fourier Domain Adaptation augmentation")
    parser.add_argument("--ema", action="store_true", help="use exponential moving average of weights")
    parser.add_argument("--lass", action="store_true",
                        help="Lesion-Aware Selective Scan: modulate SSM delta with the lesion mask")
    parser.add_argument("--lesion-pool", action="store_true",
                        help="Lesion-aware pooling: weight classification features by the lesion mask")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    # One timestamp shared by every file this run produces.
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[train] run timestamp = {TIMESTAMP}")

    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(_ROOT, args.data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data root not found: {data_root}")
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_ROOT, args.output_dir)
    run_dir = os.path.join(output_dir, f"exp_{TIMESTAMP}")
    os.makedirs(run_dir, exist_ok=True)

    # Global best weights live at output/ top level (shared across folds & runs).
    best_seg_path = os.path.join(output_dir, "best_seg.pth")
    best_cls_path = os.path.join(output_dir, "best_cls.pth")

    # Collect samples once, then stratified k-fold split.
    all_samples = BUSIDataset.collect_samples(data_root, classes=("benign", "malignant"))
    if args.exclude_prefix:
        all_samples = [s for s in all_samples
                       if not os.path.basename(s[0]).startswith(args.exclude_prefix)]
        if not all_samples:
            raise RuntimeError(f"no samples left after --exclude-prefix {args.exclude_prefix}")
    all_labels = np.array([s[2] for s in all_samples])
    print(f"[train] total samples: {len(all_samples)} (benign={int((all_labels == 0).sum())}, "
          f"malignant={int((all_labels == 1).sum())})")
    if len(all_samples) == 0:
        raise RuntimeError("no samples found — check dataset root / mask pairing")

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_splits = list(skf.split(np.zeros(len(all_samples)), all_labels))

    # Decide which folds to run.
    fold_indices = list(range(args.k_folds)) if args.fold < 0 else [args.fold]
    if args.smoke:
        fold_indices = fold_indices[:1]

    # Global best (persisted across folds at output/ top level).
    best_seg = _load_global_best(best_seg_path)  # Dice
    best_cls = _load_global_best(best_cls_path)  # ACC

    # A single resumed fold overrides the fold list and start epoch.
    resume_fold = None
    start_epoch_override = 0
    if args.resume:
        import re
        m = re.search(r"fold_(\d+)", args.resume)
        resume_fold = int(m.group(1)) if m else args.fold
        if resume_fold is None or resume_fold < 0:
            raise ValueError("--resume needs a fold in the path (e.g. fold_2) or --fold N")
        fold_indices = [resume_fold]

    fold_results = []
    for fold in fold_indices:
        tr_idx, va_idx = fold_splits[fold]
        print(f"\n[fold {fold + 1}/{args.k_folds}] train={len(tr_idx)} val={len(va_idx)} "
              f"(benign={int((all_labels[va_idx] == 0).sum())}, malignant={int((all_labels[va_idx] == 1).sum())})")

        train_ds = BUSIDataset(data_root, train=True, samples=[all_samples[i] for i in tr_idx])
        val_ds = BUSIDataset(data_root, train=False, samples=[all_samples[i] for i in va_idx])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

        # Fresh model / optimizer / scheduler per fold.
        model = _build_model(device, args.pretrained, args.lass, args.lesion_pool)
        cls_weight = None
        if args.class_weight:
            counts = np.bincount(all_labels[tr_idx]).astype(np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            w = 1.0 / counts
            w = w / w.mean()  # normalize so mean weight == 1
            cls_weight = torch.tensor(w, dtype=torch.float32).to(device)
            print(f"  [class-weight] fold {fold + 1} weights={[round(float(x), 3) for x in w]}")
        criterion_cls = nn.CrossEntropyLoss(weight=cls_weight)
        criterion_seg = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # EMA: an exponential-moving-average copy used for eval & saving.
        ema_model = None
        if args.ema:
            ema_model = copy.deepcopy(model)
            ema_model.eval()
            for p in ema_model.parameters():
                p.requires_grad_(False)

        fold_dir = os.path.join(run_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        csv_path = os.path.join(fold_dir, "train_log.csv")
        ckpt_template = os.path.join(fold_dir, "ckpt_epoch{}.tar")
        ckpt_glob = os.path.join(fold_dir, "ckpt_epoch*.tar")

        # TensorBoard writer (visualization only; not an archive).
        writer = None
        if not args.no_tb and not args.smoke:
            tb_log_dir = args.tb_dir if os.path.isabs(args.tb_dir) else os.path.join(_ROOT, args.tb_dir)
            tb_log_dir = os.path.join(tb_log_dir, f"exp_{TIMESTAMP}", f"fold_{fold}")
            writer = SummaryWriter(tb_log_dir)

        start_epoch = 0
        if args.resume and fold == resume_fold:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"[train] resumed {args.resume} at epoch {start_epoch}")

        # CSV full-history log (skipped in smoke mode to avoid leaving an empty file).
        csv_fields = ["epoch", "lr", "train_cls_loss", "train_seg_loss", "train_acc",
                      "val_cls_loss", "val_seg_loss", "val_acc", "val_auc", "val_dice", "val_iou"]
        csv_file = csv_writer = None
        if not args.smoke:
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            csv_writer.writeheader()

        max_steps = 4 if args.smoke else len(train_loader)
        fold_best_dice = fold_best_iou = 0.0
        fold_best_acc = fold_best_auc = 0.0

        for epoch in range(start_epoch, args.epochs):
            model.train()
            t0 = time.time()
            run_cls_loss = run_seg_loss = 0.0
            run_correct = run_total = 0
            for it, (images, masks, labels) in enumerate(train_loader):
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                if args.fda and np.random.rand() < 0.5:
                    images = fda(images)
                logits, mask_logits = model(images, masks) if (args.lass or args.lesion_pool) else model(images)
                loss_cls = focal_loss(logits, labels) if args.focal else criterion_cls(logits, labels)
                loss_seg = criterion_seg(mask_logits, masks)
                loss_seg = loss_seg + (tversky_loss(mask_logits, masks) if args.tversky
                                       else dice_loss(mask_logits, masks))
                loss = loss_cls + loss_seg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.ema:
                    with torch.no_grad():
                        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                            ema_p.mul_(0.995).add_(p, alpha=1.0 - 0.995)

                run_cls_loss += loss_cls.item()
                run_seg_loss += loss_seg.item()
                run_correct += (logits.argmax(1) == labels).sum().item()
                run_total += images.size(0)

                if args.smoke and it + 1 >= max_steps:
                    break

            scheduler.step()
            n_batches = max(it + 1, 1)
            train_cls_loss = run_cls_loss / n_batches
            train_seg_loss = run_seg_loss / n_batches
            train_acc = run_correct / max(run_total, 1)

            eval_model = ema_model if args.ema else model
            v_cls_loss, v_seg_loss, v_acc, v_auc, v_dice, v_iou = evaluate(
                eval_model, val_loader, device, criterion_cls, criterion_seg)
            lr = optimizer.param_groups[0]["lr"]

            print(f"[fold {fold + 1}][epoch {epoch + 1}/{args.epochs}] "
                  f"train acc={train_acc:.3f} cls={train_cls_loss:.3f} seg={train_seg_loss:.3f} | "
                  f"val acc={v_acc:.3f} auc={v_auc:.3f} dice={v_dice:.3f} iou={v_iou:.3f} ({time.time() - t0:.1f}s)")

            if args.smoke:
                break

            # --- train_log CSV ---
            csv_writer.writerow({
                "epoch": epoch + 1,
                "lr": round(lr, 6),
                "train_cls_loss": round(train_cls_loss, 6),
                "train_seg_loss": round(train_seg_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_cls_loss": round(v_cls_loss, 6),
                "val_seg_loss": round(v_seg_loss, 6),
                "val_acc": round(v_acc, 6),
                "val_auc": round(v_auc, 6),
                "val_dice": round(v_dice, 6),
                "val_iou": round(v_iou, 6),
            })
            csv_file.flush()

            # --- TensorBoard ---
            if writer is not None:
                writer.add_scalar("train/cls_loss", train_cls_loss, epoch)
                writer.add_scalar("train/seg_loss", train_seg_loss, epoch)
                writer.add_scalar("train/acc", train_acc, epoch)
                writer.add_scalar("val/cls_loss", v_cls_loss, epoch)
                writer.add_scalar("val/seg_loss", v_seg_loss, epoch)
                writer.add_scalar("val/acc", v_acc, epoch)
                writer.add_scalar("val/auc", v_auc, epoch)
                writer.add_scalar("val/dice", v_dice, epoch)
                writer.add_scalar("val/iou", v_iou, epoch)
                writer.add_scalar("lr", lr, epoch)

            # --- fold best (for the CV summary) ---
            if v_dice > fold_best_dice:
                fold_best_dice, fold_best_iou = v_dice, v_iou
            if v_acc > fold_best_acc:
                fold_best_acc, fold_best_auc = v_acc, v_auc

            # --- global best weights (only overwrite when beating the global best) ---
            if v_dice > best_seg:
                best_seg = v_dice
                torch.save({"model": eval_model.state_dict(), "metric": v_dice}, best_seg_path)
                print(f"  [saved] best_seg (global Dice={v_dice:.3f})")
            if v_acc > best_cls:
                best_cls = v_acc
                torch.save({"model": eval_model.state_dict(), "metric": v_acc}, best_cls_path)
                print(f"  [saved] best_cls (global acc={v_acc:.3f})")

            # --- periodic resume checkpoint + cleanup (this fold only) ---
            if (epoch + 1) % args.save_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, ckpt_template.format(epoch + 1))
                print(f"  [saved] ckpt epoch {epoch + 1}")
                self_ckpts = sorted(glob.glob(ckpt_glob), key=_ckpt_epoch)
                for old in self_ckpts[:-args.max_ckpt]:
                    os.remove(old)
                    print(f"  [removed] {os.path.basename(old)}")

        if csv_file is not None:
            csv_file.close()
        if writer is not None:
            writer.close()

        if not args.smoke:
            fold_results.append({
                "fold": fold + 1,
                "best_dice": fold_best_dice,
                "best_iou": fold_best_iou,
                "best_acc": fold_best_acc,
                "best_auc": fold_best_auc,
            })
            print(f"[fold {fold + 1}] best dice={fold_best_dice:.3f} iou={fold_best_iou:.3f} "
                  f"acc={fold_best_acc:.3f} auc={fold_best_auc:.3f}")

    # --- CV summary across folds ---
    if fold_results:
        summary_path = os.path.join(run_dir, "cv_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold", "best_dice", "best_iou", "best_acc", "best_auc"])
            w.writeheader()
            for r in fold_results:
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
        dice = np.array([r["best_dice"] for r in fold_results])
        iou = np.array([r["best_iou"] for r in fold_results])
        acc = np.array([r["best_acc"] for r in fold_results])
        auc = np.array([r["best_auc"] for r in fold_results])
        print("\n===== cross-validation summary =====")
        for r in fold_results:
            print(f"  fold {r['fold']}: dice={r['best_dice']:.3f} iou={r['best_iou']:.3f} "
                  f"acc={r['best_acc']:.3f} auc={r['best_auc']:.3f}")
        print(f"  MEAN: dice={dice.mean():.3f}±{dice.std():.3f}  iou={iou.mean():.3f}±{iou.std():.3f}  "
              f"acc={acc.mean():.3f}±{acc.std():.3f}  auc={auc.mean():.3f}±{auc.std():.3f}")
        print(f"  saved -> {summary_path}")

    print(f"\n[train] done. global best_seg={best_seg:.3f} (Dice), best_cls={best_cls:.3f} (acc)")


if __name__ == "__main__":
    main()

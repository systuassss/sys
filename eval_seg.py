"""Evaluate the multi-task model's SEGMENTATION head on the masked test sets.

Uses the real masks already gathered in dataset/multitask_all (busi_* / uc_* /
uclm_* have ground-truth masks). private_date has no mask, so it is skipped.
Reports per-image Dice and IoU (mean).

Usage:
    python eval_seg.py --weights output_busbra/best_seg.pth
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from multitask.model import MultiTaskVMamba, VSSM_TINY
from multitask.dataset import IMAGENET_MEAN, IMAGENET_STD

MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
SIZE = 224

PREFIXES = {"BUSI": "busi_", "BUS_UC": "uc_", "BUS_UCLM": "uclm_"}


def collect(root, prefix):
    """Return [(img_path, mask_path)] from {benign,malignant} dirs matching prefix."""
    pairs = []
    for cls in ("benign", "malignant"):
        d = os.path.join(root, cls)
        for f in sorted(os.listdir(d)):
            if f.endswith(".png") and not f.endswith("_mask.png") and f.startswith(prefix):
                img = os.path.join(d, f)
                mask = os.path.join(d, f.replace(".png", "_mask.png"))
                if os.path.exists(mask):
                    pairs.append((img, mask))
    return pairs


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


def load_mask(path):
    m = Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    return (np.asarray(m, dtype=np.float32) / 255.0 > 0.5).astype(np.uint8)


def dice_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    uni = (pred | gt).sum()
    d = 2 * inter / (union + 1e-6)
    i = inter / (uni + 1e-6)
    return float(d), float(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.weights, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    use_lass = any("lesion_dt" in k for k in state.keys())
    use_lesion_pool = any("lesion_scale" in k for k in state.keys())
    model = MultiTaskVMamba(encoder_config=VSSM_TINY, num_classes=2,
                            use_lass=use_lass, use_lesion_pool=use_lesion_pool)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[eval-seg] loaded {args.weights} (metric={ck.get('metric', float('nan')) if isinstance(ck, dict) else 'na'})")

    root = os.path.join(_ROOT, "dataset", "multitask_all")
    for name, prefix in PREFIXES.items():
        pairs = collect(root, prefix)
        if not pairs:
            print(f"{name}: no samples")
            continue
        ds = []  # (img_path, mask_path)
        ds = pairs
        dice_list, iou_list = [], []
        with torch.no_grad():
            for i in range(0, len(ds), args.batch_size):
                batch = ds[i:i + args.batch_size]
                x = torch.stack([preprocess(p[0]) for p in batch]).to(device)
                x = (x - MEAN.to(device)) / STD.to(device)
                _, mask_logits = model(x)
                preds = (torch.sigmoid(mask_logits[:, 0]).cpu().numpy() > 0.5).astype(np.uint8)
                for (_, mp), pr in zip(batch, preds):
                    gt = load_mask(mp)
                    d, iou = dice_iou(pr, gt)
                    dice_list.append(d)
                    iou_list.append(iou)
        print(f"{'=' * 56}")
        print(f"Test set (seg): {name}   ({len(ds)} images)")
        print(f"  Dice: {np.mean(dice_list):.4f}  (std {np.std(dice_list):.4f})")
        print(f"  IoU : {np.mean(iou_list):.4f}  (std {np.std(iou_list):.4f})")
        print()


if __name__ == "__main__":
    main()

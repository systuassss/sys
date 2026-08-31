"""Segmentation inference demo.

Loads output/best_seg.pth and runs the shared encoder + seg decoder on tumor
images, then saves a side-by-side figure: input / ground truth / prediction /
overlay (red=prediction, green=ground truth, yellow=overlap).

Usage:
  python multitask/infer_seg.py                          # auto-pick a few malignant images
  python multitask/infer_seg.py --image path.png --mask path_mask.png   # your own image
  python multitask/infer_seg.py --n 6 --out output/seg_demo
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from multitask.model import MultiTaskVMamba, VSSM_TINY
from multitask.dataset import IMAGENET_MEAN, IMAGENET_STD, BUSIDataset

MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
SIZE = 224


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr.transpose(2, 0, 1))
    t = (t - MEAN) / STD
    return t.unsqueeze(0), img


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


def overlay(orig, gt, pred):
    out = np.asarray(orig).astype(np.float32).copy()
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    # green = ground truth, red = prediction, yellow = overlap
    out[gt_b, 0] *= 0.4
    out[gt_b, 1] = 255
    out[gt_b, 2] *= 0.4
    out[pred_b, 0] = 255
    out[pred_b, 1] *= 0.4
    out[pred_b, 2] *= 0.4
    both = gt_b & pred_b
    out[both, 0] = 255
    out[both, 1] = 255
    out[both, 2] = 0
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(_ROOT, "output", "best_seg.pth"))
    ap.add_argument("--data-root", default="dataset/Dataset_BUSI_with_GT")
    ap.add_argument("--image", default="")
    ap.add_argument("--mask", default="")
    ap.add_argument("--n", type=int, default=4, help="number of auto-picked images")
    ap.add_argument("--out", default="output/seg_demo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt["model"]
    use_lass = any("lesion_dt" in k for k in state.keys())
    use_lesion_pool = any("lesion_scale" in k for k in state.keys())
    model = MultiTaskVMamba(encoder_config=VSSM_TINY, num_classes=2,
                            use_lass=use_lass, use_lesion_pool=use_lesion_pool)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[infer] loaded {args.weights} (metric={ckpt.get('metric'):.4f}) -> {device}")

    # Build the list of (image, mask) to run.
    pairs = []
    if args.image:
        mask = args.mask or args.image.replace(".png", "_mask.png")
        pairs.append((args.image, mask))
    else:
        data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(_ROOT, args.data_root)
        samples = BUSIDataset.collect_samples(data_root, classes=("malignant",))
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(samples), size=min(args.n, len(samples)), replace=False)
        pairs = [(samples[i][0], samples[i][1]) for i in idx]

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    for k, (img_path, mask_path) in enumerate(pairs):
        t, orig = preprocess(img_path)
        gt = load_mask(mask_path) if os.path.exists(mask_path) else np.zeros((SIZE, SIZE), np.uint8)
        with torch.no_grad():
            _, mask_logits = model(t.to(device))
        pred = (torch.sigmoid(mask_logits[0, 0]).cpu().numpy() > 0.5).astype(np.uint8)

        d, i = dice_iou(pred, gt) if gt.any() else (float("nan"), float("nan"))
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{k + 1}/{len(pairs)}] {name}: dice={d:.3f} iou={i:.3f}")

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(orig); axes[0].set_title("Input"); axes[0].axis("off")
        axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground truth"); axes[1].axis("off")
        axes[2].imshow(pred, cmap="gray"); axes[2].set_title("Prediction"); axes[2].axis("off")
        axes[3].imshow(overlay(orig, gt, pred)); axes[3].set_title("Overlay (red=pred, green=GT)"); axes[3].axis("off")
        fig.suptitle(f"{name}   dice={d:.3f}  iou={i:.3f}")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{name}_seg.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"        saved -> {out_path}")

    print(f"[infer] done. figures in {out_dir}")


if __name__ == "__main__":
    main()

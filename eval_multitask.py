"""Evaluate the multi-task model's CLASSIFICATION head on the 4 test sets.

The model returns (logits, mask); we use only `logits` here. Optional horizontal
flip TTA. Metrics: accuracy, binary AUC, precision, recall, specificity, F1.

Usage:
    python eval_multitask.py --weights output_mixed/best_cls.pth \
        --test-sets dataset/Dataset_BUSI_bm dataset/private_date \
                      dataset/Dataset_BUS_UC_bm dataset/Dataset_BUS_UCLM_bm
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from multitask.model import MultiTaskVMamba, VSSM_TINY
from multitask.dataset import IMAGENET_MEAN, IMAGENET_STD


def get_eval_transform():
    # Matches the multitask training preprocessing (resize to 224, ImageNet norm).
    return transforms.Compose([
        transforms.Resize((224, 224), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def predict(model, loader, device, tta=True):
    model.eval()
    logits = []
    for images, _ in loader:
        images = images.to(device)
        out = model(images)[0]  # (logits, mask)
        if tta:
            out_flip = model(torch.flip(images, dims=[3]))[0]
            out = (out + out_flip) / 2.0
        logits.append(out.cpu())
    return torch.cat(logits, dim=0)


def report(name, logits, targets):
    probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(1)
    y = targets.numpy()
    acc = accuracy_score(y, pred) * 100
    auc = roc_auc_score(y, probs[:, 1]) * 100 if len(np.unique(y)) > 1 else float("nan")
    prec = precision_score(y, pred, zero_division=0) * 100
    rec = recall_score(y, pred, zero_division=0) * 100
    f1 = f1_score(y, pred, zero_division=0) * 100
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else float("nan")
    print(f"{'=' * 62}")
    print(f"Test set: {name}")
    print(f"{'=' * 62}")
    print(f"  Accuracy:    {acc:.2f}%")
    print(f"  AUC:         {auc:.2f}%")
    print(f"  Precision:   {prec:.2f}%")
    print(f"  Recall:      {rec:.2f}%")
    print(f"  Specificity: {spec:.2f}%")
    print(f"  F1-Score:    {f1:.2f}%")
    print(f"  Confusion [[TN,FP],[FN,TP]]: [[{tn},{fp}],[{fn},{tp}]]")
    print()
    return {"acc": acc, "auc": auc, "prec": prec, "rec": rec, "spec": spec, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to a MultiTaskVMamba checkpoint")
    ap.add_argument("--test-sets", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskVMamba(encoder_config=VSSM_TINY, num_classes=2)
    ck = torch.load(args.weights, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[eval] loaded {args.weights} (metric={ck.get('metric', float('nan')) if isinstance(ck, dict) else 'na'})")

    all_acc = []
    for path in args.test_sets:
        ds = ImageFolder(path)
        ds.transform = get_eval_transform()
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        targets = torch.tensor(ds.targets)
        logits = predict(model, loader, device, tta=not args.no_tta)
        r = report(path, logits, targets)
        all_acc.append(r["acc"])

    if all_acc:
        print(f"===== MEAN accuracy over {len(all_acc)} test sets: "
              f"{np.mean(all_acc):.2f}% =====")


if __name__ == "__main__":
    main()

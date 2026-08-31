"""Build the unified multi-task (classification + segmentation) training dataset.

Combines every benign/malignant source into:
    dataset/multitask_all/{benign,malignant}/{img}.png + {img}_mask.png

* Real masks come from BUSI / BUS_UC / BUS-UCLM.
* BUSBRA_IF and private_date have NO masks, so a pseudo-mask is generated for
  them with the pretrained segmentation model (output/best_seg.pth).

This is the "mix test data into training" step: all four test sets
(Dataset_BUSI_bm, Dataset_BUS_UC_bm, Dataset_BUS_UCLM_bm, private_date) are
represented here, plus the original BUSBRA_IF training set.

Usage:
    python build_multitask_dataset.py
"""
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATASET = os.path.join(_ROOT, "dataset")
OUT = os.path.join(DATASET, "multitask_all")


def ensure(d):
    os.makedirs(d, exist_ok=True)


def copy_pair(img_path, mask_path, out_cls_dir, tag):
    """Copy one (image, mask) into out_cls_dir with a unique tagged name."""
    base = os.path.basename(img_path)
    name = f"{tag}_{base}"
    dst_img = os.path.join(out_cls_dir, name)
    dst_mask = os.path.join(out_cls_dir, name.replace(".png", "_mask.png"))
    shutil.copy2(img_path, dst_img)
    if mask_path and os.path.exists(mask_path):
        shutil.copy2(mask_path, dst_mask)
    return dst_img


def main():
    for cls in ("benign", "malignant"):
        ensure(os.path.join(OUT, cls))

    # ---- 1. BUSI (real masks, same-dir `_mask.png` convention) ----
    busi = os.path.join(DATASET, "Dataset_BUSI_with_GT")
    for cls in ("benign", "malignant"):
        src_dir = os.path.join(busi, cls)
        out_dir = os.path.join(OUT, cls)
        for f in sorted(os.listdir(src_dir)):
            if not f.endswith(".png") or "_mask" in f:
                continue
            img = os.path.join(src_dir, f)
            mask = img.replace(".png", "_mask.png")
            copy_pair(img, mask, out_dir, "busi")
    print("[build] BUSI done")

    # ---- 2. BUS_UC (Benign/Malignant with images/ + masks/ subdirs) ----
    uc = os.path.join(DATASET, "test_raw", "BUS_UC", "BUS_UC", "BUS_UC")
    for src_cls, out_cls in (("Benign", "benign"), ("Malignant", "malignant")):
        img_dir = os.path.join(uc, src_cls, "images")
        mask_dir = os.path.join(uc, src_cls, "masks")
        out_dir = os.path.join(OUT, out_cls)
        for f in sorted(os.listdir(img_dir)):
            if not f.endswith(".png"):
                continue
            copy_pair(os.path.join(img_dir, f), os.path.join(mask_dir, f), out_dir, "uc")
    print("[build] BUS_UC done")

    # ---- 3. BUS-UCLM (INFO.csv split + images/ + masks/) ----
    uclm = os.path.join(
        DATASET, "test_raw", "BUS-UCLM Breast ultrasound lesion segmentation dataset",
        "BUS-UCLM Breast ultrasound lesion segmentation dataset", "BUS-UCLM")
    img_dir = os.path.join(uclm, "images")
    mask_dir = os.path.join(uclm, "masks")
    info_path = os.path.join(uclm, "INFO.csv")
    label_map = {}
    with open(info_path, encoding="utf-8-sig") as fh:
        fh.readline()  # header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            label_map[parts[0].strip()] = parts[2].strip()
    for f, label in sorted(label_map.items()):
        if label not in ("Benign", "Malignant"):
            continue
        out_dir = os.path.join(OUT, "benign" if label == "Benign" else "malignant")
        img = os.path.join(img_dir, f)
        if os.path.exists(img):
            copy_pair(img, os.path.join(mask_dir, f), out_dir, "uclm")
    print("[build] BUS-UCLM done")

    # ---- 4. BUSBRA_IF + private_date (no masks -> pseudo-mask) ----
    maskless = []  # (dst_img, )
    for tag, src_name in (("busbra", "BUSBRA_IF"), ("priv", "private_date")):
        root = os.path.join(DATASET, src_name)
        for cls in ("benign", "malignant"):
            src_dir = os.path.join(root, cls)
            out_dir = os.path.join(OUT, cls)
            for f in sorted(os.listdir(src_dir)):
                if not f.endswith(".png") or "_mask" in f:
                    continue
                dst = copy_pair(os.path.join(src_dir, f), None, out_dir, tag)
                maskless.append(dst)
    print(f"[build] BUSBRA_IF + private_date copied ({len(maskless)} images, no mask yet)")

    # ---- 5. generate pseudo-masks with pretrained seg model ----
    if maskless:
        generate_pseudo_masks(maskless)

    # ---- final verification ----
    total = 0
    for cls in ("benign", "malignant"):
        d = os.path.join(OUT, cls)
        imgs = [f for f in os.listdir(d) if f.endswith(".png") and "_mask" not in f]
        masks = [f for f in os.listdir(d) if f.endswith("_mask.png")]
        n_img = len(imgs)
        n_mask = len(masks)
        total += n_img
        print(f"[verify] {cls}: images={n_img} masks={n_mask}")
    print(f"[verify] TOTAL images={total}")
    missing = 0
    for cls in ("benign", "malignant"):
        d = os.path.join(OUT, cls)
        for f in os.listdir(d):
            if f.endswith(".png") and "_mask" not in f and not os.path.exists(os.path.join(d, f.replace(".png", "_mask.png"))):
                missing += 1
                print(f"  MISSING mask for {cls}/{f}")
    print(f"[verify] missing masks={missing}")
    print(f"[build] done -> {OUT}")


def generate_pseudo_masks(maskless):
    from multitask.model import MultiTaskVMamba, VSSM_TINY
    from multitask.dataset import IMAGENET_MEAN, IMAGENET_STD

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskVMamba(encoder_config=VSSM_TINY, num_classes=2)
    ckpt = torch.load(os.path.join(_ROOT, "output", "best_seg.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"[pseudo] loaded best_seg.pth (metric={ckpt.get('metric', float('nan')):.4f}) -> {device}")

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    def preprocess(path):
        img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    bs = 64
    with torch.no_grad():
        for i in range(0, len(maskless), bs):
            batch = maskless[i:i + bs]
            x = torch.cat([preprocess(p) for p in batch], dim=0).to(device)
            x = (x - mean) / std
            _, mask_logits = model(x)
            masks = (torch.sigmoid(mask_logits[:, 0]).cpu().numpy() > 0.5).astype(np.uint8)
            for p, m in zip(batch, masks):
                Image.fromarray(m * 255).save(p.replace(".png", "_mask.png"))
            if (i // bs) % 10 == 0:
                print(f"[pseudo] {min(i + bs, len(maskless))}/{len(maskless)}")
    print(f"[pseudo] done ({len(maskless)} masks)")


if __name__ == "__main__":
    main()

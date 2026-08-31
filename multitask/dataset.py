"""BUSI dataset loader for multi-task (classification + segmentation).

Uses only the benign and malignant classes (skips normal, per project scope).
Each sample returns (image, mask, label); the mask is binary (lesion=1).

Augmentation is done on tensors (vectorized) rather than PIL so it stays fast
enough to feed the GPU even with a small number of workers.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _elastic_displacement(h, w, sigma=6.0, grid=8):
    """Random smooth displacement field (1, h, w, 2) for elastic deformation."""
    disp = torch.randn(2, grid, grid) * sigma
    disp = torch.nn.functional.interpolate(
        disp.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)  # (2, h, w)
    return disp.permute(1, 2, 0).unsqueeze(0)  # (1, h, w, 2)


class BUSIDataset(Dataset):
    def __init__(self, root, classes=("benign", "malignant"), size=224, train=True, samples=None):
        self.root = root
        self.classes = list(classes)
        self.size = size
        self.train = train
        self.samples = samples if samples is not None else self.collect_samples(root, classes)
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    @staticmethod
    def collect_samples(root, classes=("benign", "malignant")):
        """Return [(image_path, mask_path, label), ...] for the given classes."""
        samples = []
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in sorted(os.listdir(cls_dir)):
                if not f.endswith(".png") or "_mask" in f:
                    continue
                img_path = os.path.join(cls_dir, f)
                mask_path = img_path.replace(".png", "_mask.png")
                if os.path.exists(mask_path):
                    samples.append((img_path, mask_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize both to a common size first (still PIL; cheap).
        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        # Convert to float tensors once, then run all augmentation vectorized.
        image = TF.to_tensor(image)  # (3, H, W) float [0, 1]
        mask = TF.to_tensor(mask)    # (1, H, W) float [0, 1]

        # Ultrasound-specific augmentation (training only). Spatial transforms are
        # applied to image + mask together so they stay aligned; intensity/colour
        # transforms touch the image only.
        if self.train:
            # 1) random grayscale: strip the device-specific colormap.
            if np.random.rand() < 0.5:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

            # 2) random affine (rotation / scale / translation / shear).
            angle = float(np.random.uniform(-15, 15))
            scale = float(np.random.uniform(0.85, 1.15))
            tx = float(np.random.uniform(-0.10, 0.10) * self.size)
            ty = float(np.random.uniform(-0.10, 0.10) * self.size)
            shear = float(np.random.uniform(-10, 10))
            image = TF.affine(image, angle, (tx, ty), scale, shear,
                              interpolation=InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.affine(mask, angle, (tx, ty), scale, shear,
                             interpolation=InterpolationMode.NEAREST, fill=0.0)

            # 3) random horizontal flip.
            if np.random.rand() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 4) intensity jitter (brightness / contrast / gamma) on image only.
            image = TF.adjust_brightness(image, float(np.random.uniform(0.7, 1.3)))
            image = TF.adjust_contrast(image, float(np.random.uniform(0.7, 1.3)))
            image = TF.adjust_gamma(image, float(np.random.uniform(0.7, 1.3)))

        # ImageNet normalization; binary mask.
        image = (image - self.mean) / self.std
        mask = (mask > 0.5).float()

        return image, mask, torch.tensor(label, dtype=torch.long)

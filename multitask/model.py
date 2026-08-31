"""Multi-task VMamba model.

Shared VMamba backbone + classification head (benign/malignant) + segmentation
decoder (lesion vs background). The backbone is the official VMamba-T ("0230")
architecture so it can load the released ImageNet-pretrained checkpoint.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the BU-Mamba repo root importable so we can reach the vendored VMamba code.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VMamba.classification.models.vmamba import Backbone_VSSM
from VMamba.classification.models.csms6s import CrossScan


# Official VMamba-T (v2, "0230") config — matches vmambav2_tiny_224.yaml.
VSSM_TINY = dict(
    patch_size=4,
    in_chans=3,
    depths=[2, 2, 5, 2],
    dims=96,
    ssm_d_state=1,
    ssm_ratio=2.0,
    ssm_dt_rank="auto",
    ssm_act_layer="silu",
    ssm_conv=3,
    ssm_conv_bias=False,
    ssm_drop_rate=0.0,
    ssm_init="v0",
    forward_type="v05_noz",
    mlp_ratio=4.0,
    mlp_act_layer="gelu",
    mlp_drop_rate=0.0,
    drop_path_rate=0.2,
    patch_norm=True,
    norm_layer="ln2d",
    downsample_version="v3",
    patchembed_version="v2",
    gmlp=False,
    use_checkpoint=False,
    posembed=False,
    imgsize=224,
)


class ClsHead(nn.Module):
    """Classification head: (lesion-aware) global average pooling + linear."""

    def __init__(self, in_dim, num_classes, use_lesion=False):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_dim, num_classes)
        self.use_lesion = use_lesion
        # Learnable scalar: how much to up-weight lesion regions (init 0 => no-op).
        self.lesion_scale = nn.Parameter(torch.zeros(1)) if use_lesion else None

    def forward(self, x, mask=None):
        if self.use_lesion and mask is not None:
            m = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x * (1.0 + self.lesion_scale * m)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SegDecoder(nn.Module):
    """U-Net style decoder: upsamples the deepest feature with skip connections."""

    def __init__(self, dims, out_channels=1):
        super().__init__()
        d0, d1, d2, d3 = dims  # 96, 192, 384, 768

        self.up3 = nn.ConvTranspose2d(d3, d2, kernel_size=2, stride=2)
        self.dec3 = self._block(d2 * 2, d2)
        self.up2 = nn.ConvTranspose2d(d2, d1, kernel_size=2, stride=2)
        self.dec2 = self._block(d1 * 2, d1)
        self.up1 = nn.ConvTranspose2d(d1, d0, kernel_size=2, stride=2)
        self.dec1 = self._block(d0 * 2, d0)
        self.up0 = nn.ConvTranspose2d(d0, d0 // 2, kernel_size=2, stride=2)
        self.up_final = nn.ConvTranspose2d(d0 // 2, d0 // 2, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(d0 // 2, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        f0, f1, f2, f3 = feats  # spatial sizes 56, 28, 14, 7
        x = self.up3(f3)                          # 7 -> 14
        x = self.dec3(torch.cat([x, f2], dim=1))   # concat skip f2
        x = self.up2(x)                            # 14 -> 28
        x = self.dec2(torch.cat([x, f1], dim=1))   # concat skip f1
        x = self.up1(x)                            # 28 -> 56
        x = self.dec1(torch.cat([x, f0], dim=1))   # concat skip f0
        x = self.up0(x)                            # 56 -> 112
        x = self.up_final(x)                       # 112 -> 224
        return self.out_conv(x)                    # (B, 1, 224, 224)


class MultiTaskVMamba(nn.Module):
    """Shared VMamba backbone + classification head + segmentation decoder.

    Optional LASS (Lesion-Aware Selective Scan): modulate the SSM S6 delta (dt)
    with the lesion mask so the scan focuses on lesion regions, making the
    classification features more domain-invariant. Enabled with use_lass=True;
    during training the GT mask is passed, at inference the model's own mask.
    """

    def __init__(self, encoder_config=VSSM_TINY, num_classes=2, pretrained=None,
                 use_lass=False, use_lesion_pool=False):
        super().__init__()
        config = dict(encoder_config)
        config.pop("norm_layer", None)  # Backbone_VSSM sets norm_layer explicitly
        self.encoder = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            norm_layer="ln2d",
            **config,
        )
        self.dims = list(self.encoder.dims)  # [96, 192, 384, 768]
        self.num_classes = num_classes
        self.use_lesion_pool = use_lesion_pool
        self.cls_head = ClsHead(self.dims[-1], num_classes, use_lesion=use_lesion_pool)
        self.seg_head = SegDecoder(self.dims, out_channels=1)

        self.use_lass = use_lass
        self._lass_dt = None  # per-stage (B, 4*D, L) delta modulation cache
        if use_lass:
            # Per-stage learnable scalar: how strongly the lesion mask shifts dt.
            # Init 0 => pure no-op at start (safe warm-up).
            self.lesion_dt = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(len(self.dims))]
            )
            # d_inner per stage, read from the actual SS2D blocks (ssm_ratio * dim).
            self._d_inner = []
            for i in range(len(self.dims)):
                op = self.encoder.layers[i].blocks[0].op
                self._d_inner.append(int(op.dt_projs_weight.shape[1]))
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def _build_dt_bias(self, mask):
        """mask: (B, 1, 224, 224). Return per-stage (B, 4*d_inner, L) dt biases
        aligned to the cross-scan order used inside forward_corev2."""
        B = mask.size(0)
        sizes = [56, 28, 14, 7]  # spatial size each SS2D stage sees
        dt_bias = []
        for i, s in enumerate(sizes):
            L = s * s
            mi = F.interpolate(mask, size=(s, s), mode="bilinear", align_corners=False)
            ms = CrossScan.apply(mi)  # (B, 4, 1, L) in cross-scan order
            d = self._d_inner[i]
            dt_bias.append((self.lesion_dt[i] * ms).expand(B, 4, d, L).reshape(B, 4 * d, L))
        return dt_bias

    def load_pretrained(self, path):
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[MultiTaskVMamba] WARNING: failed to load {path}: {e}")
            print("  training from scratch (random init)")
            return
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        # Strip the ImageNet classification head; the encoder has none.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"[MultiTaskVMamba] loaded pretrained backbone from {path}")
        print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    def forward(self, x, mask=None):
        if not (self.use_lass or self.use_lesion_pool):
            feats = self.encoder(x)          # list of 4 feature maps (B, C, H, W)
            logits = self.cls_head(feats[-1])   # (B, num_classes)
            seg = self.seg_head(feats)          # (B, 1, H, W)
            return logits, seg

        # Need a lesion mask: predict one first if none is supplied (two-pass).
        if mask is None:
            with torch.no_grad():
                feats0 = self.encoder(x)
                mask = torch.sigmoid(self.seg_head(feats0))
        dt_bias = self._build_dt_bias(mask) if self.use_lass else None
        feats = self.encoder(x, dt_bias=dt_bias)
        logits = self.cls_head(feats[-1], mask=mask if self.use_lesion_pool else None)
        return logits, self.seg_head(feats)

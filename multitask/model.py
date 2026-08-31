"""Multi-task VMamba model.

Shared VMamba backbone + classification head (benign/malignant) + segmentation
decoder (lesion vs background). The backbone is the official VMamba-T ("0230")
architecture so it can load the released ImageNet-pretrained checkpoint.
"""
import os
import sys

import torch
import torch.nn as nn

# Make the BU-Mamba repo root importable so we can reach the vendored VMamba code.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VMamba.classification.models.vmamba import Backbone_VSSM


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
    """Classification head: global average pooling + linear."""

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
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
    """Shared VMamba backbone + classification head + segmentation decoder."""

    def __init__(self, encoder_config=VSSM_TINY, num_classes=2, pretrained=None):
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
        self.cls_head = ClsHead(self.dims[-1], num_classes)
        self.seg_head = SegDecoder(self.dims, out_channels=1)
        if pretrained is not None:
            self.load_pretrained(pretrained)

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

    def forward(self, x):
        feats = self.encoder(x)          # list of 4 feature maps (B, C, H, W)
        logits = self.cls_head(feats[-1])   # (B, num_classes)
        mask = self.seg_head(feats)         # (B, 1, H, W)
        return logits, mask

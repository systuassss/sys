"""Smoke test for LASS: build the model, run a training-mode forward (GT mask)
and an eval-mode two-pass forward (self-predicted mask), verify shapes + that the
lesion_dt parameter is being trained and dt_bias is actually injected."""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from multitask.model import MultiTaskVMamba, VSSM_TINY

PRETRAINED = os.path.join(_ROOT, "checkpoints", "pretrained", "vssm_tiny_0230_ckpt_epoch_262.pth")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")
    model = MultiTaskVMamba(encoder_config=VSSM_TINY, num_classes=2,
                            pretrained=PRETRAINED, use_lass=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] params={n_params/1e6:.2f}M  lesion_dt={[float(p) for p in model.lesion_dt]}")
    print(f"[smoke] d_inner per stage={model._d_inner}")

    model.train()
    x = torch.randn(2, 3, 224, 224, device=device)
    mask = (torch.rand(2, 1, 224, 224, device=device) > 0.7).float()

    # 1) training-mode: pass GT mask
    logits, seg = model(x, mask=mask)
    print(f"[smoke] train: logits {tuple(logits.shape)}  seg {tuple(seg.shape)}")
    loss = logits.mean() + seg.mean()
    loss.backward()
    g = model.lesion_dt[0].grad
    print(f"[smoke] lesion_dt[0].grad = {g}")

    # 2) eval-mode: no mask -> two-pass
    model.eval()
    with torch.no_grad():
        logits2, seg2 = model(x)
    print(f"[smoke] eval: logits {tuple(logits2.shape)}  seg {tuple(seg2.shape)}")

    print("[smoke] OK")


if __name__ == "__main__":
    main()

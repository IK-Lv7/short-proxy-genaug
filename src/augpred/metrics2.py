"""Training-free extra baselines requested in review (Major 6).

- precision / recall (Kynkäänniemi et al. 2019, k=3) and density / coverage (Naeem et al. 2020, k=5): Inception features
- Class-conditional KID: mean of per-class KID (unbiased MMD^2 with polynomial kernel). Contrast to FID, which ignores labels
- Fréchet distance on DINOv2 features (Stein et al. 2023): CLS features of facebook/dinov2-base
"""
import numpy as np
import torch
import torch.nn.functional as F


def _knn_radius(x, k):
    d = torch.cdist(x, x)
    return d.topk(k + 1, largest=False).values[:, -1]  # k-th distance excluding self (distance 0)


@torch.no_grad()
def prdc(real, fake, k_pr=3, k_dc=5, device="cuda"):
    """real, fake: (N, d) features. Returns precision / recall / density / coverage."""
    r = torch.as_tensor(real, device=device, dtype=torch.float32)
    f = torch.as_tensor(fake, device=device, dtype=torch.float32)
    d_rf = torch.cdist(r, f)  # (Nr, Nf)
    rad_r3, rad_f3 = _knn_radius(r, k_pr), _knn_radius(f, k_pr)
    precision = (d_rf <= rad_r3[:, None]).any(0).float().mean().item()
    recall = (d_rf.T <= rad_f3[:, None]).any(0).float().mean().item()
    rad_r5 = _knn_radius(r, k_dc)
    inside = d_rf <= rad_r5[:, None]
    density = (inside.sum(0).float() / k_dc).mean().item()
    coverage = (d_rf.min(1).values <= rad_r5).float().mean().item()
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


def _poly_mmd2_unbiased(x, y):
    d = x.shape[1]
    kxx = (x @ x.T / d + 1) ** 3
    kyy = (y @ y.T / d + 1) ** 3
    kxy = (x @ y.T / d + 1) ** 3
    m, n = len(x), len(y)
    return ((kxx.sum() - kxx.diag().sum()) / (m * (m - 1)) + (kyy.sum() - kyy.diag().sum()) / (n * (n - 1))
            - 2 * kxy.mean()).item()


@torch.no_grad()
def class_conditional_kid(real, y_real, fake, y_fake, device="cuda"):
    r = torch.as_tensor(real, device=device, dtype=torch.float64)
    f = torch.as_tensor(fake, device=device, dtype=torch.float64)
    y_real, y_fake = np.asarray(y_real), np.asarray(y_fake)
    vals = [_poly_mmd2_unbiased(r[y_real == c], f[y_fake == c]) for c in np.unique(y_fake)
            if (y_real == c).sum() > 1 and (y_fake == c).sum() > 1]
    return float(np.mean(vals))


_DINO = None


@torch.no_grad()
def dinov2_features(x_uint8, device="cuda", batch_size=128, name="facebook/dinov2-base", cache_dir="data/hf_cache"):
    """(N,3,H,W) uint8 -> (N, 768) CLS features. Preprocessing is DINOv2 standard (resize and crop to 224px)."""
    global _DINO
    from transformers import AutoImageProcessor, AutoModel
    if _DINO is None:
        model = AutoModel.from_pretrained(name, cache_dir=cache_dir, use_safetensors=True).to(device).eval().half()
        _DINO = (model, AutoImageProcessor.from_pretrained(name, cache_dir=cache_dir))
    model, proc = _DINO
    out = []
    for i in range(0, len(x_uint8), batch_size):
        imgs = list(x_uint8[i:i + batch_size].permute(0, 2, 3, 1).numpy())
        pix = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device).half()
        out.append(model(pixel_values=pix).last_hidden_state[:, 0].float().cpu())
    return torch.cat(out).numpy()

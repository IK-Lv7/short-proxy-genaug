"""Metrics computable from a set of generated images without training.

Every metric takes the set of generated images actually added to training under that condition (x_gen, y_gen).
The real images used for comparison are the training subset (5000 images); test is not used.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import CIFAR100_MEAN, CIFAR100_STD


# ---- Inception features (same preprocessing as clean-fid) ----

class _CleanFIDDataset(Dataset):
    def __init__(self, x_uint8_nchw):
        from cleanfid.resize import build_resizer
        self.x = x_uint8_nchw.permute(0, 2, 3, 1).numpy()
        self.fn_resize = build_resizer("clean")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.fn_resize(self.x[i])  # float32 (299, 299, 3), range [0, 255]
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


_INCEPTION = None


@torch.no_grad()
def inception_features(x, device="cuda", batch_size=200, num_workers=8):
    global _INCEPTION
    from cleanfid.features import build_feature_extractor
    if _INCEPTION is None:
        _INCEPTION = build_feature_extractor("clean", device=torch.device(device), use_dataparallel=False)
    loader = DataLoader(_CleanFIDDataset(x), batch_size, num_workers=num_workers)
    return np.concatenate([_INCEPTION(b.to(device)).cpu().numpy() for b in loader])


def fid_from_features(f1, f2):
    from cleanfid.fid import frechet_distance
    return float(frechet_distance(f1.mean(0), np.cov(f1, rowvar=False), f2.mean(0), np.cov(f2, rowvar=False)))


def kid_from_features(f1, f2):
    from cleanfid.fid import kernel_distance
    return float(kernel_distance(f1, f2))


# ---- Metrics using a trained classifier ----

@torch.no_grad()
def classifier_outputs(model, x, device="cuda", batch_size=1000):
    """(N,3,32,32) uint8 -> (logits, features)"""
    model.eval()
    mean = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=device).view(1, 3, 1, 1)
    logits, feats = [], []
    for i in range(0, len(x), batch_size):
        xb = (x[i:i + batch_size].to(device).float() / 255 - mean) / std
        f = model.features(xb)
        feats.append(f.float().cpu())
        logits.append(model.fc(f).float().cpu())
    return torch.cat(logits), torch.cat(feats)


def label_metrics(logits, y):
    """Label consistency: fraction classified as the assigned label by the oracle classifier, and mean probability of the assigned label."""
    y = torch.as_tensor(y)
    p = logits.softmax(1)
    return {"acc": (logits.argmax(1) == y).float().mean().item(),
            "prob": p[torch.arange(len(y)), y].mean().item()}


def within_class_diversity(feats, y):
    """Mean pairwise cosine distance within each class (averaged over classes)."""
    f = F.normalize(feats, dim=1)
    y = np.asarray(y)
    vals = []
    for c in np.unique(y):
        fc = f[y == c]
        if len(fc) < 2:
            continue
        sim = fc @ fc.T
        n = len(fc)
        vals.append(1 - (sim.sum() - n).item() / (n * (n - 1)))
    return float(np.mean(vals))


def novelty(feats_gen, feats_src):
    """Mean cosine distance between generated images and their source real images."""
    return float((1 - (F.normalize(feats_gen, dim=1) * F.normalize(feats_src, dim=1)).sum(1)).mean())

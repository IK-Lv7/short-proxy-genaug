"""Multi-class version of Discriminative Span (Desai & Narendra, arXiv 2605.09697).

The original is for binary classification (positives synthesized from real negatives). Ported here as follows:
  1. Use CLIP image embeddings (L2-normalized)
  2. Train multi-class logistic regression on embeddings of real images (training subset) to get class weights w_c
  3. Build matrix D_c of differences between embeddings of class-c generated images and their source real images
  4. Recover w_c as a linear combination of rows of D_c; relative error RPE = ||w_c - D_c^T a|| / ||w_c||, DS_c = 1 - RPE
  5. Average over classes
Recovery uses three solvers: least squares (lstsq), ridge, and non-negative least squares (NNLS). The original paper does not give the ridge strength,
so lambda = 0.1 * mean(diag(D_c D_c^T)) was fixed in advance.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls

_CLIP = None


@torch.no_grad()
def clip_embed(x_uint8, device="cuda", batch_size=256, name="openai/clip-vit-base-patch32", cache_dir="data/hf_cache"):
    """(N,3,H,W) uint8 -> (N, d) L2-normalized CLIP image embeddings"""
    global _CLIP
    from transformers import CLIPImageProcessor, CLIPModel
    if _CLIP is None:
        model = CLIPModel.from_pretrained(name, cache_dir=cache_dir).to(device).eval().half()
        _CLIP = (model, CLIPImageProcessor.from_pretrained(name, cache_dir=cache_dir))
    model, proc = _CLIP
    out = []
    for i in range(0, len(x_uint8), batch_size):
        imgs = list(x_uint8[i:i + batch_size].permute(0, 2, 3, 1).numpy())
        pix = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device).half()
        f = model.get_image_features(pixel_values=pix)
        if not torch.is_tensor(f):  # some transformers versions return an output object
            f = f.pooler_output if getattr(f, "pooler_output", None) is not None else f[0]
        out.append(F.normalize(f.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def fit_linear(emb, y, num_classes, epochs=300, wd=1e-4, device="cuda"):
    """Multi-class logistic regression (full-batch L-BFGS). Returns weights (C, d)."""
    x = torch.as_tensor(emb, device=device)
    t = torch.as_tensor(y, device=device)
    W = torch.zeros(num_classes, x.shape[1], device=device, requires_grad=True)
    b = torch.zeros(num_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ W.T + b, t) + wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    acc = ((x @ W.T + b).argmax(1) == t).float().mean().item()
    return W.detach().cpu().numpy(), acc


def _rpe(w, D, solver):
    if len(D) == 0:
        return 1.0
    if solver == "lstsq":
        a = np.linalg.lstsq(D.T, w, rcond=None)[0]
    elif solver == "ridge":
        G = D @ D.T
        lam = 0.1 * np.mean(np.diag(G))
        a = np.linalg.solve(G + lam * np.eye(len(D)), D @ w)
    elif solver == "nnls":
        a = nnls(D.T, w)[0]
    else:
        raise ValueError(solver)
    return float(np.linalg.norm(w - D.T @ a) / np.linalg.norm(w))


def discriminative_span(W, emb_gen, emb_src, y_gen, solvers=("lstsq", "ridge", "nnls")):
    """Return class-averaged DS = 1 - RPE for each solver."""
    diff = emb_gen - emb_src
    y_gen = np.asarray(y_gen)
    out = {}
    for s in solvers:
        vals = [1 - _rpe(W[c], diff[y_gen == c], s) for c in range(len(W))]
        out[f"ds_{s}"] = float(np.mean(vals))
    return out

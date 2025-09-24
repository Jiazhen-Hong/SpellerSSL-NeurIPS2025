from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from typing import Tuple, Dict
import numpy as np
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import norm


CHARA_MAP = {
    'A': (7,1), 'B': (7,2), 'C': (7,3), 'D': (7,4), 'E': (7,5), 'F': (7,6),
    'G': (8,1), 'H': (8,2), 'I': (8,3), 'J': (8,4), 'K': (8,5), 'L': (8,6),
    'M': (9,1), 'N': (9,2), 'O': (9,3), 'P': (9,4), 'Q': (9,5), 'R': (9,6),
    'S': (10,1),'T': (10,2),'U': (10,3),'V': (10,4),'W': (10,5),'X': (10,6),
    'Y': (11,1),'Z': (11,2),'1': (11,3),'2': (11,4),'3': (11,5),'4': (11,6),
    '5': (12,1),'6': (12,2),'7': (12,3),'8': (12,4),'9': (12,5),'_': (12,6)
}


class P300Dataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, t_slice: Tuple[int,int] | None = None):
        assert X.ndim == 3 and y.ndim == 1 and X.shape[0] == y.shape[0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.t_slice = t_slice  # (t0, t1) or None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.t_slice is not None:
            t0, t1 = self.t_slice
            x = x[:, t0:t1]
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

# IO 
def load_splits(npz_dir: str, subject: str, agg_size: int, sfreq: int):
    path = os.path.join(npz_dir, f"{subject}_splits_g{agg_size}_sf{sfreq}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    data = np.load(path, allow_pickle=True)
    X_train = data["X_train"]; Y_train = data["Y_train"]
    X_val   = data["X_val"];   Y_val   = data["Y_val"]
    X_test  = data["X_test"];  Y_test  = data["Y_test"]
    X_test_char = data["X_test_char"]  # for char-level eval
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test, X_test_char)

def _build_t_slice(args, T_total):
    if args.t_start_ms is None or args.t_end_ms is None:
        return None
    assert args.t_end_ms > args.t_start_ms, "t_end_ms must > t_start_ms"
    t0 = int(round(args.t_start_ms / 1000.0 * args.sfreq))
    t1 = int(round(args.t_end_ms   / 1000.0 * args.sfreq))
    t0 = max(0, min(t0, T_total))
    t1 = max(0, min(t1, T_total))
    assert t1 > t0, f"bad slice [{t0},{t1})"
    return (t0, t1)

def apply_tslice_np(X: np.ndarray, t_slice):
    if t_slice is None:
        return X
    t0, t1 = t_slice
    return X[:, :, t0:t1]

# Eval
def evaluate(model, loader, device, ce_loss: nn.Module):
    model.eval()
    all_y, all_pred, all_prob1, all_logits = [], [], [], []
    loss_total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)                       # [B, 2]
            loss = ce_loss(logits, yb)
            loss_total += loss.item() * xb.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            pred  = torch.argmax(logits, dim=1)

            all_y.append(yb.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_prob1.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y         = np.concatenate(all_y, axis=0)
    pred      = np.concatenate(all_pred, axis=0)
    prob1     = np.concatenate(all_prob1, axis=0)
    logits_all= np.concatenate(all_logits, axis=0)
    acc = accuracy_score(y, pred)
    f1  = f1_score(y, pred)
    try:
        auroc = roc_auc_score(y, prob1)
    except ValueError:
        auroc = float('nan')
    loss_avg = loss_total / len(y)
    return ({"loss": loss_avg, "acc": acc, "f1": f1, "auroc": auroc}, pred, prob1, logits_all, y)

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    n = len(y)
    c0 = max(1, (y == 0).sum())
    c1 = max(1, (y == 1).sum())
    w0 = n / (2.0 * c0)
    w1 = n / (2.0 * c1)
    return torch.tensor([w0, w1], dtype=torch.float32)

def char_level_accuracy(prob1_test: np.ndarray, chars_test: np.ndarray) -> float:
    assert prob1_test.ndim == 1 and chars_test.ndim == 1
    N = prob1_test.shape[0]
    assert N % 12 == 0, "test length must be multiple of 12"
    G = N // 12
    probs = prob1_test.reshape(G, 12)
    chars = chars_test.reshape(G, 12)[:, 0]
    top2_idx = np.argsort(-probs, axis=1)[:, :2]
    pred_sets = [set(list(idx + 1)) for idx in top2_idx]
    gt_sets = [set(CHARA_MAP[str(ch)]) for ch in chars]
    hits = [int(pred_sets[i] == gt_sets[i]) for i in range(G)]
    return float(np.mean(hits))

def bandpass_filter_batch(X: np.ndarray, sfreq: float,
                          l_freq: float = 0.1, h_freq: float = 10.0,
                          order: int = 4) -> np.ndarray:
    if l_freq is None and h_freq is None:
        return X
    if l_freq is None:
        Wn, btype = h_freq, 'lowpass'
    elif h_freq is None:
        Wn, btype = l_freq, 'highpass'
    else:
        Wn, btype = [l_freq, h_freq], 'bandpass'
    sos = butter(order, Wn, btype=btype, fs=sfreq, output='sos')
    N, C, T = X.shape
    X2 = X.reshape(N * C, T)
    Xf = sosfiltfilt(sos, X2, axis=-1)
    return Xf.reshape(N, C, T)

def baseline_first100ms(X: np.ndarray, sfreq: float, ms: int = 100) -> np.ndarray:
    n = int(round(ms / 1000.0 * sfreq))
    if n <= 0 or n > X.shape[-1]:
        return X
    base = X[..., :n].mean(axis=-1, keepdims=True)
    return X - base

def format_for_latex(res):
    return " & ".join([f"{int(v)}" for v in res])

def character_accuracy_over_reps(model, X_test, Ch_test, device,
                                 reps_per_char=15, group_size=2,
                                 score_type="prob", temperature=1.0):
    N_total, C, T = X_test.shape
    assert N_total % 12 == 0, "test length must be multiple of 12"
    num_groups_per_char = reps_per_char - group_size + 1
    N_groups = N_total // 12
    assert N_groups % num_groups_per_char == 0
    n_instances = N_groups // num_groups_per_char
    Xg  = X_test.reshape(N_groups, 12, C, T)
    Chg = Ch_test.reshape(N_groups, 12)[:, 0]
    Xi  = Xg.reshape(n_instances, num_groups_per_char, 12, C, T)
    Chi = Chg.reshape(n_instances, num_groups_per_char)[:, 0]

    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(n_instances):
            x = Xi[i].reshape(num_groups_per_char * 12, C, T)
            xb = torch.from_numpy(x.astype(np.float32)).to(device)
            out = model(xb)  # [G*12, 2]
            if score_type == "prob":
                s = torch.softmax(out / temperature, dim=1)[:, 1].cpu().numpy()
            elif score_type == "logodds":
                s = (out[:, 1] - out[:, 0]).cpu().numpy()
            elif score_type == "logit":
                s = out[:, 1].cpu().numpy()
            else:
                raise ValueError(f"Unknown score_type {score_type}")
            scores.append(s.reshape(num_groups_per_char, 12))

    scores = np.stack(scores, axis=0)  # [I, G, 12]
    accs = []
    for n in range(1, num_groups_per_char + 1):
        S = scores[:, :n, :].sum(axis=1)      # [I,12]
        cols_scores = S[:, :6]
        rows_scores = S[:, 6:]
        pred_c = cols_scores.argmax(axis=1) + 1
        pred_r = rows_scores.argmax(axis=1) + 7
        right = 0
        for i in range(n_instances):
            gt_r, gt_c = CHARA_MAP[str(Chi[i])]
            if pred_c[i] == gt_c and pred_r[i] == gt_r:
                right += 1
        acc = right / n_instances
        accs.append(acc)
        print(f"n={n:2d} | Char Acc ({score_type}) = {acc:.4f}")
    return accs

def char_preview_12cols(model, Xte, ChTest, t_slice, args, device):
    from __main__ import preprocess_linear
    num_groups_per_char = args.reps_per_char - args.agg_test + 1
    idx = num_groups_per_char * 12
    xb_np = preprocess_linear(
        Xte[:idx],
        t_slice=t_slice,
        ma_win=args.ma_win,
        decim=args.decim,
        sfreq=args.sfreq,
        lp_hz=getattr(args, "lp_hz", None)
    ).astype(np.float32)
    xb = torch.from_numpy(xb_np).to(device)
    logits = model(xb)
    logits1 = logits[:, 1].cpu().numpy().reshape(num_groups_per_char, 12)
    print("\nlogit_1:")
    np.set_printoptions(precision=4, suppress=True)
    print(logits1)
    logodds = (logits[:, 1] - logits[:, 0]).cpu().numpy().reshape(num_groups_per_char, 12)
    print("\nlog-odds:")
    print(logodds)
    probs1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().reshape(num_groups_per_char, 12)
    print("\nprob y=1:")
    print(probs1)
    chars = ChTest[:idx].reshape(num_groups_per_char, 12)[:, 0]
    print("\nchar labels:")
    print(chars)

def plot_binary_score_distribution(
    scores_or_logits,
    labels,
    save_path="score_dist.png",
    bins=30,
    xlim=None,
    ylim=None,
    title="Score Distributions (P300 vs Non-P300)"
):
    plt.rcParams.update({
        "font.size": 16,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    if isinstance(scores_or_logits, torch.Tensor):
        scores_or_logits = scores_or_logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    arr = np.asarray(scores_or_logits)
    if arr.ndim == 2 and arr.shape[1] == 2:
        scores = arr[:, 1] - arr[:, 0]   # log-odds
    elif arr.ndim == 1:
        scores = arr                     # already 1D scores
    else:
        raise ValueError("scores_or_logits must be shape [N] or [N,2].")

    labels = labels.astype(int)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("labels must be 0/1.")

    s1 = scores[labels == 1]  # P300
    s0 = scores[labels == 0]  # Non-P300

    mu1, sd1 = float(np.mean(s1)), float(np.std(s1) + 1e-12)
    mu0, sd0 = float(np.mean(s0)), float(np.std(s0) + 1e-12)
    fdr = (mu1 - mu0) ** 2 / (sd1 ** 2 + sd0 ** 2)

    xmin = np.min(scores) if xlim is None else xlim[0]
    xmax = np.max(scores) if xlim is None else xlim[1]
    pad = 0.05 * (xmax - xmin + 1e-9)
    xs = np.linspace(xmin - pad, xmax + pad, 500)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.plot(xs, norm.pdf(xs, loc=mu1, scale=sd1),
            color="green", linewidth=3.0, label="P300 PDF")
    ax.plot(xs, norm.pdf(xs, loc=mu0, scale=sd0),
            color="black", linestyle="--", linewidth=3.0, label="Non-P300 PDF")

    n1, b1, p1 = ax.hist(s1, bins=bins, density=True, alpha=0.35,
                         color="green", edgecolor="black", linewidth=0.8)
    for patch in p1:
        patch.set_hatch("//")
    ax.hist(s0, bins=bins, density=True, alpha=0.35,
            color="black", edgecolor="black", linewidth=0.8)

    ax.text(0.02, 0.95, rf"$\mu_0$ = {mu0:.2f}\n" + rf"$\sigma_0$ = {sd0:.2f}",
            transform=ax.transAxes, fontsize=20, color="black",
            ha="left", va="top")
    ax.text(0.02, 0.80, rf"$\mu_1$ = {mu1:.2f}\n" + rf"$\sigma_1$ = {sd1:.2f}",
            transform=ax.transAxes, fontsize=20, color="green",
            ha="left", va="top")
    ax.text(0.98, 0.50, rf"FDR = {fdr:.2f}",
            transform=ax.transAxes, fontsize=20, color="black",
            ha="right", va="bottom")

    ax.set_xlabel("Scores", fontsize=18, fontweight="bold")
    ax.set_ylabel("Density", fontsize=18, fontweight="bold")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)

    handles = [
        Line2D([0],[0], color="green", linewidth=3.0, label="P300 PDF"),
        Patch(facecolor="green", edgecolor="black", alpha=0.35, hatch="//", label="P300 Hist"),
        Line2D([0],[0], color="black", linewidth=3.0, linestyle="--", label="Non-P300 PDF"),
        Patch(facecolor="black", edgecolor="black", alpha=0.35, label="Non-P300 Hist"),
    ]
    leg = ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=18)
    for text in leg.get_texts():
        text.set_fontweight('bold')

    ax.grid(alpha=0.25, linewidth=1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")
    return fdr
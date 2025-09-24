# Fine.py
import os
import argparse
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utility.dataprepare import (
    P300Dataset, apply_tslice_np, load_splits, compute_class_weights,
    char_level_accuracy, _build_t_slice, evaluate, character_accuracy_over_reps,
    format_for_latex, plot_binary_score_distribution, char_preview_12cols
)
from utility.Downstream import finetune_unet1d_cls


def print_header(title: str):
    bar = "═" * (len(title) + 2)
    print(f"\n{bar}\n {title}\n{bar}")

def print_kv(key: str, value: str, indent: int = 2):
    print(" " * indent + f"{key}: {value}")

def print_shape(name: str, arr: np.ndarray, indent: int = 2):
    print_kv(name, f"{tuple(arr.shape)}", indent)


# ---------- load multiple train/val aggregations; single test aggregation ----------
def load_mixed_data(data_dir: str, subject: str, sfreq: int,
                    agg_list: List[int], agg_test: int):
    Xtr_all, Ytr_all, Xva_all, Yva_all = [], [], [], []
    counts = []
    for g in agg_list:
        (Xtr_g, Ytr_g), (Xva_g, Yva_g), _ = load_splits(data_dir, subject, g, sfreq)
        Xtr_all.append(Xtr_g); Ytr_all.append(Ytr_g)
        Xva_all.append(Xva_g); Yva_all.append(Yva_g)
        counts.append((g, len(Xtr_g), len(Xva_g)))

    Xtr = np.concatenate(Xtr_all, axis=0)
    Ytr = np.concatenate(Ytr_all, axis=0)
    Xva = np.concatenate(Xva_all, axis=0)
    Yva = np.concatenate(Yva_all, axis=0)

    (_, _), (_, _), (Xte, Yte, ChTest) = load_splits(data_dir, subject, agg_test, sfreq)

    print_header("Load mixed Train/Val and single Test aggregation")
    print_kv("Train/Val aggregations", f"{agg_list}")
    for g, n_tr, n_va in counts:
        print_kv(f"g={g:<2d}", f"train={n_tr:<6d} | val={n_va:<6d}", indent=4)
    print_kv("Test aggregation", f"g={agg_test}")
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte, ChTest)


# ---------- slice first K chars (optional) ----------
def _slice_first_k_chars(X: np.ndarray, y: np.ndarray, k_chars: int,
                         reps_per_char: int, agg_size: int) -> Tuple[np.ndarray, np.ndarray]:
    num_per_char = (reps_per_char - agg_size + 1) * 12
    if num_per_char <= 0:
        raise ValueError("num_per_char invalid.")
    n_take = min(X.shape[0], k_chars * num_per_char)
    return X[:n_take], y[:n_take]

def build_ft_subset(
    data_dir: str, subject: str, sfreq: int, agg_list: List[int], *,
    k_chars: int, reps_per_char: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int,int,int]]]:
    Xtr_all, Ytr_all, Xva_all, Yva_all = [], [], [], []
    details = []
    for g in agg_list:
        (Xtr_g, Ytr_g), (Xva_g, Yva_g), _ = load_splits(data_dir, subject, g, sfreq)
        Xtr_s, Ytr_s = _slice_first_k_chars(Xtr_g, Ytr_g, k_chars, reps_per_char, g)
        Xva_s, Yva_s = _slice_first_k_chars(Xva_g, Yva_g, k_chars, reps_per_char, g)
        Xtr_all.append(Xtr_s); Ytr_all.append(Ytr_s)
        Xva_all.append(Xva_s); Yva_all.append(Yva_s)
        details.append((g, len(Xtr_s), len(Xva_s)))
    Xtr_ft = np.concatenate(Xtr_all, axis=0) if Xtr_all else np.empty((0,))
    Ytr_ft = np.concatenate(Ytr_all, axis=0) if Ytr_all else np.empty((0,))
    Xva_ft = np.concatenate(Xva_all, axis=0) if Xva_all else np.empty((0,))
    Yva_ft = np.concatenate(Yva_all, axis=0) if Yva_all else np.empty((0,))
    return Xtr_ft, Ytr_ft, Xva_ft, Yva_ft, details


# ---------- pad time to multiple of 16 ----------
def pad_time_to_multiple_of_16(X: np.ndarray) -> np.ndarray:
    N, C, T = X.shape
    rem = T % 16
    if rem == 0:
        return X
    pad = 16 - rem
    Xp = np.pad(X, ((0,0),(0,0),(0,pad)), mode="constant")
    print_kv("Pad to x16", f"T {T} -> {Xp.shape[-1]} (+{pad})", indent=4)
    return Xp


def preprocess_linear(X: np.ndarray, t_slice=None, args=None):
    return apply_tslice_np(X, t_slice) if t_slice is not None else X


# ---------- main ----------
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    (Xtr_full, Ytr_full), (Xva_full, Yva_full), (Xte, Yte, ChTest) = load_mixed_data(
        args.data_dir, args.subject, args.sfreq, args.agg_list, args.agg_test
    )

    # optional subset by first K chars
    use_ft_subset = (args.ft_chars is not None and int(args.ft_chars) != 0)
    if use_ft_subset:
        print_header(f"Build fine-tune subset by first {args.ft_chars} chars per agg (Train-first, then Val)")
        Xtr_sel_list, Ytr_sel_list = [], []
        K_req = int(args.ft_chars)

        for g in args.agg_list:
            (Xtr_g, Ytr_g), (Xva_g, Yva_g), _ = load_splits(args.data_dir, args.subject, g, args.sfreq)

            num_per_char = (args.reps_per_char - g + 1) * 12
            train_chars  = len(Ytr_g) // num_per_char
            val_chars    = len(Yva_g) // num_per_char
            total_chars  = train_chars + val_chars

            if K_req <= 0 or K_req >= total_chars:
                X_take = np.concatenate([Xtr_g, Xva_g], axis=0)
                Y_take = np.concatenate([Ytr_g, Yva_g], axis=0)
                print_kv(f"g={g:<2d}", f"use FULL: train={train_chars}, val={val_chars} (chars={total_chars})", indent=4)

            elif K_req <= train_chars:
                n_tr = K_req * num_per_char
                X_take = Xtr_g[:n_tr]
                Y_take = Ytr_g[:n_tr]
                print_kv(f"g={g:<2d}", f"train-only: take {K_req}/{train_chars} chars -> {n_tr} samples", indent=4)

            else:
                k_res = K_req - train_chars
                n_tr  = train_chars * num_per_char
                n_va  = k_res * num_per_char
                X_take = np.concatenate([Xtr_g[:n_tr], Xva_g[:n_va]], axis=0)
                Y_take = np.concatenate([Ytr_g[:n_tr], Yva_g[:n_va]], axis=0)
                print_kv(f"g={g:<2d}", f"train {train_chars} + val {k_res} chars -> {n_tr+n_va} samples", indent=4)

            Xtr_sel_list.append(X_take)
            Ytr_sel_list.append(Y_take)

        Xtr = np.concatenate(Xtr_sel_list, axis=0)
        Ytr = np.concatenate(Ytr_sel_list, axis=0)

    else:
        Xtr = np.concatenate([Xtr_full, Xva_full], axis=0)
        Ytr = np.concatenate([Ytr_full, Yva_full], axis=0)

    # time crop
    C, T = Xtr.shape[1], Xtr.shape[2]
    t_slice = _build_t_slice(args, T)

    print_header("Time window / cropping")
    if t_slice is not None:
        s_ms = t_slice[0] / args.sfreq * 1000.0
        e_ms = t_slice[1] / args.sfreq * 1000.0
        print_kv("Use samples", f"{t_slice}  (~ {s_ms:.0f}–{e_ms:.0f} ms)")
        Xtr = apply_tslice_np(Xtr, t_slice)
        Xte = apply_tslice_np(Xte, t_slice)
    else:
        print_kv("Use samples", "FULL length (no crop)")

    # pad T to /16
    print_header("Pad to 16-multiple for UNet1D")
    Xtr = pad_time_to_multiple_of_16(Xtr)
    Xte = pad_time_to_multiple_of_16(Xte)

    C, T_model = Xtr.shape[1], Xtr.shape[2]
    print_header("Shapes after crop/pad")
    print_shape("Train X", Xtr)
    print_shape("Test  X", Xte)
    print_kv("Channels (C)", f"{C}")
    print_kv("Time (T_model)", f"{T_model}  (T_model%16={T_model%16})")

    # loaders
    class P300DatasetPad(P300Dataset):
        pass

    train_ds = P300DatasetPad(Xtr, Ytr, t_slice=None)
    test_ds  = P300DatasetPad(Xte, Yte, t_slice=None)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=RandomSampler(train_ds),
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, sampler=SequentialSampler(test_ds),
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # class balance print
    print_header("Class balance (train)")
    cls_w = compute_class_weights(Ytr)
    pos_ratio = float(Ytr.mean()); neg_ratio = 1.0 - pos_ratio
    print_kv("pos_ratio", f"{pos_ratio:.4f}")
    print_kv("suggested CE weights", f"{cls_w.numpy().round(4).tolist()}")
    print_kv("imbalance", f"{neg_ratio/pos_ratio:.2f}:1 (neg:pos)")

    # output dir
    tag_ft = f"_ft{args.ft_chars}" if use_ft_subset else "_ftAll"
    result_dir = os.path.join(
        "./results_paper",
        f"sub-{args.subject}_gtr-{'-'.join(map(str,args.agg_list))}_gte-{args.agg_test}_{tag_ft}"
    )
    os.makedirs(result_dir, exist_ok=True)
    print_header("Output")
    print_kv("result_dir", result_dir)
    print_kv("ckpt_ssl", args.ckpt_ssl if args.ckpt_ssl else "(scratch)")

    # fine-tune (no val)
    print_header("Fine-tuning (Lightning, train+val merged)")
    lit, test_metrics = finetune_unet1d_cls(
        ckpt_path=args.ckpt_ssl or None,
        train_loader=train_loader,
        val_loader=None,
        test_loader=test_loader,
        result_dir=result_dir,
        epochs=args.epochs_ft,
        seed=args.seed,
        lr=args.lr, max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout_cls,
        freeze_encoder=args.freeze_encoder
    )

    print_header("Final Test (Lightning reported)")
    if len(test_metrics) > 0:
        for k, v in test_metrics[0].items():
            print_kv(k, f"{float(v):.6f}")

    # eval again with utility.evaluate
    print_header("Trial-level metrics (utility.evaluate)")
    class _WrapLit(torch.nn.Module):
        def __init__(self, lit_module):
            super().__init__()
            self.lit = lit_module
        def forward(self, x):
            return self.lit.model(x)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    wrap_model = _WrapLit(lit).to(device).eval()

    ce_loss = nn.CrossEntropyLoss(weight=cls_w.to(device))
    tr_metrics, _, _, _, _ = evaluate(wrap_model, train_loader, device, ce_loss)
    te_metrics, te_pred, te_prob1, logits1, ytest = evaluate(wrap_model, test_loader, device, ce_loss)

    print_kv("Train", f"Loss {tr_metrics['loss']:.4f} | Acc {tr_metrics['acc']:.4f} | F1 {tr_metrics['f1']:.4f} | AUROC {tr_metrics['auroc']:.4f}")
    print_kv("Test ", f"Loss {te_metrics['loss']:.4f} | Acc {te_metrics['acc']:.4f} | F1 {te_metrics['f1']:.4f} | AUROC {te_metrics['auroc']:.4f}")

    # char-level acc (top-2 columns)
    print_header("Character-level accuracy (top-2 columns match)")
    char_acc = char_level_accuracy(te_prob1, ChTest)
    print_kv("Acc_char", f"{char_acc:.4f}")

    # score distribution (saves FDR)
    save_png = f"./Figure/Logit_figure/logits_dist_{args.subject}_{args.agg_list}{tag_ft}_noVal.png"
    os.makedirs(os.path.dirname(save_png), exist_ok=True)
    fdr = plot_binary_score_distribution(logits1, ytest, save_path=save_png)

    # repetition-wise char acc
    print_header("Character Accuracy over repetitions (accumulation)")
    Xte_crop = apply_tslice_np(Xte, t_slice) if t_slice is not None else Xte
    acc_list = character_accuracy_over_reps(
        wrap_model, Xte_crop, ChTest, device,
        reps_per_char=args.reps_per_char,
        group_size=args.agg_test,
        score_type="logit",
        temperature=2
    )
    res = np.array(acc_list) * 100
    print(format_for_latex(res))
    print(",".join([f"{int(v)}" for v in res]))

    f1 = te_metrics['f1']
    return {"seed": args.seed, "f1": f1, "fdr": float(fdr)}


# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, default="/data/Aggregation",
                   help="contains {SUBJECT}_splits_g{g}_sf{sfreq}.npz")
    p.add_argument("--subject", type=str, default="B", choices=["A", "B", "C"])
    p.add_argument("--agg_list", type=int, nargs="+", default=[2], help="train/val aggregation size list")
    p.add_argument("--agg_test", type=int, default=1, help="test aggregation size")
    p.add_argument("--sfreq", type=int, default=240)

    # time crop
    p.add_argument("--t_start_ms", type=int, default=0)
    p.add_argument("--t_end_ms",   type=int, default=667)

    # finetune
    p.add_argument("--ckpt_ssl", type=str, default="./Checkpoint/Pretraining/Aall_667.ckpt")
    p.add_argument("--epochs_ft", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout_cls", type=float, default=0)
    p.add_argument("--freeze_encoder", action="store_true")

    # loader / env
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # eval
    p.add_argument("--reps_per_char", type=int, default=15)
    p.add_argument("--ft_chars", type=int, default=85)  # 0 uses all Train+Val

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
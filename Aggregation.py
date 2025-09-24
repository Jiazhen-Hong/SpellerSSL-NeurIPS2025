import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from src.BCI_P300_II2b_III3a import get_data_III

# Simple ERP plot
def plot_erp(X, y, sfreq, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    t = np.arange(X.shape[2]) / sfreq
    target_avg = X[y == 1].mean(axis=0)
    nontarget_avg = X[y == 0].mean(axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(t, target_avg[10] + 1.2, label="Target", linewidth=2, color="red")
    plt.plot(t, nontarget_avg[10], label="Non-target", linewidth=2, color="black")
    plt.xlabel("Time (s)", fontsize=16, weight="bold")
    plt.ylabel("Amplitude (uV)", fontsize=16, weight="bold")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved figure: {save_path}")


CHAR_MAP = {
    "A": (7, 1), "B": (7, 2), "C": (7, 3), "D": (7, 4), "E": (7, 5), "F": (7, 6),
    "G": (8, 1), "H": (8, 2), "I": (8, 3), "J": (8, 4), "K": (8, 5), "L": (8, 6),
    "M": (9, 1), "N": (9, 2), "O": (9, 3), "P": (9, 4), "Q": (9, 5), "R": (9, 6),
    "S": (10, 1), "T": (10, 2), "U": (10, 3), "V": (10, 4), "W": (10, 5), "X": (10, 6),
    "Y": (11, 1), "Z": (11, 2), "1": (11, 3), "2": (11, 4), "3": (11, 5), "4": (11, 6),
    "5": (12, 1), "6": (12, 2), "7": (12, 3), "8": (12, 4), "9": (12, 5), "_": (12, 6),
}


# Main
if __name__ == "__main__":
    subject_id = "B"
    split = "train"
    sfreq = 240
    window_len = 0.8
    group_size = 4

    # Load data
    raw, stim_code, targets = get_data_III(subject_id, split, path="/data/BCIC_P300_RAW/III2a")
    stim_code = stim_code.squeeze(0)

    # Epochs
    from utility.Aggregation import (
        extract_epochs,
        crossalign_grouped,
        flatten_groups,
        print_groups_for_char,
    )

    X, y = extract_epochs(
        raw, stim_channel="stim", event_dict=None, epoch_len=window_len, sfreq=sfreq
    )
    print(f"Original: X={X.shape}, y={y.shape}")
    plot_erp(
        X,
        y,
        sfreq=sfreq,
        title="P300 Response",
        save_path=f"./Figure/{subject_id}_{split}_ERP_{sfreq}Hz.png",
    )
    print("--" * 30)

    # Align to canonical [1..12], then group by g
    Xg, yg = crossalign_grouped(X, stim_code, targets, group_size=group_size)
    print(f"Grouped (canonical 1..12): Xg={Xg.shape}, yg={yg.shape}")

    # Quick check of first few chars
    chars_to_check = [0, 1]
    for cidx in chars_to_check:
        print_groups_for_char(
            yg,
            targets,
            char_idx=cidx,
            group_size=group_size,
            reps_per_char=15,
            max_groups=None,
        )

    # Flatten for a simple plot
    X_new, y_new = flatten_groups(Xg, yg)  # [N, g*12, C, T], [N, g*12]
    print(f"Flattened: X={X_new.shape}, y={y_new.shape}")

    X_plot = X_new.reshape(-1, X_new.shape[2], X_new.shape[3])
    y_plot = y_new.reshape(-1)
    plot_erp(
        X_plot,
        y_plot,
        sfreq=sfreq,
        title=f"Cross-Aligned ERP (g={group_size})",
        save_path=f"./Figure/{subject_id}_{split}_crossaligned_ERP_{sfreq}Hz.png",
    )
    print("--" * 30)

    # Mean over reps inside each group
    X_mean = Xg.mean(axis=1)                 # [N, 12, C, T]
    y_mean = (yg.sum(axis=1) > 0).astype(np.int32)  # [N, 12]

    print(f"After mean-aggregation: X_mean={X_mean.shape}, y_mean={y_mean.shape}")

    # Meta per group
    reps_per_char = 15
    num_groups_per_char = reps_per_char - group_size + 1
    N_groups = Xg.shape[0]

    char_list = []
    rowcols = []
    for g in range(N_groups):
        char_idx = g // num_groups_per_char
        ch = targets[char_idx]
        char_list.append(ch)
        rowcols.append(CHAR_MAP[ch])

    char_list = np.array(char_list)               # [N]
    rowcols = np.array(rowcols, dtype=np.int32)   # [N, 2]

    print("char_list sample:", char_list[:10])
    print("rowcols sample:", rowcols[:10])

    # Plot again with mean-agg
    X_plot = X_mean.reshape(-1, X_mean.shape[2], X_mean.shape[3])
    y_plot = y_mean.reshape(-1)
    plot_erp(
        X_plot,
        y_plot,
        sfreq=sfreq,
        title=f"Cross-Aligned ERP (g={group_size}, mean-agg)",
        save_path=f"./Figure/{subject_id}_{split}_crossalignedMean_ERP_{sfreq}Hz.png",
    )

    # Save dataset
    save_dir = "/data/BCICP300/raw/Aggregation/Data_without_reshape"
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(
        save_dir, f"{subject_id}_{split}_g{group_size}_sfreq{sfreq}.npz"
    )
    np.savez_compressed(
        npz_path,
        X=X_mean,            # [N, 12, C, T]
        y=y_mean,            # [N, 12]
        chars=char_list,     # [N]
        rowcols=rowcols,     # [N, 2]
        sfreq=sfreq,
        window_len=window_len,
        group_size=group_size,
        reps_per_char=reps_per_char,
    )
    print(f"Saved aggregated dataset: {npz_path}")
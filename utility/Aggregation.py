import mne
import numpy as np

chara_map = {
    'A': (7, 1), 'B': (7, 2), 'C': (7, 3), 'D': (7, 4), 'E': (7, 5), 'F': (7, 6),
    'G': (8, 1), 'H': (8, 2), 'I': (8, 3), 'J': (8, 4), 'K': (8, 5), 'L': (8, 6),
    'M': (9, 1), 'N': (9, 2), 'O': (9, 3), 'P': (9, 4), 'Q': (9, 5), 'R': (9, 6),
    'S': (10,1), 'T': (10,2), 'U': (10,3), 'V': (10,4), 'W': (10,5), 'X': (10,6),
    'Y': (11,1), 'Z': (11,2), '1': (11,3), '2': (11,4), '3': (11,5), '4': (11,6),
    '5': (12,1), '6': (12,2), '7': (12,3), '8': (12,4), '9': (12,5), '_': (12,6)
}


def extract_epochs(raw, stim_channel='stim', event_dict=None, epoch_len=1.0, sfreq=128):
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq)
    events = mne.find_events(raw, stim_channel=stim_channel, initial_event=True)
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=0, tmax=epoch_len,
                        baseline=None, detrend=1, picks='eeg', preload=True, verbose=False)
    X = epochs.get_data()[:, :, :-1]  # remove last point from MNE
    y = epochs.events[:, -1] - 1      # convert labels to 0/1
    return X, y  


def flatten_groups(Xg, yg):
    N, g, twelve, C, T = Xg.shape
    return Xg.reshape(N, g*twelve, C, T), yg.reshape(N, g*twelve)


def extract_stimulus_cycles_from_code(stim_code_row):
    flashes, cycle_list, count = [], [], 0
    idx = 0
    while idx < len(stim_code_row):
        val = int(stim_code_row[idx])
        if val != 0:
            flashes.append(val)
            count += 1
            while idx + 1 < len(stim_code_row) and int(stim_code_row[idx + 1]) == val:
                idx += 1
            if count == 12:
                cycle_list.append(flashes)
                flashes, count = [], 0
        idx += 1
    return cycle_list  # len==15, each sublist has len==12


def crossalign_grouped(X, stim_code, targets, group_size=2):
    assert X.shape[0] % 180 == 0, "total trials are not 180 multiple"
    n_chars = X.shape[0] // 180

    X_groups, y_groups = [], []
    for char_idx, ch in enumerate(targets[:n_chars]):
        char_start = char_idx * 180
        X_char = X[char_start:char_start+180]  # [180,C,T]

        cycles = extract_stimulus_cycles_from_code(stim_code[char_idx])
        assert len(cycles) == 15 and all(len(c)==12 for c in cycles), "cycles are not 15 x 12"
        X_ord = reorder_trials_to_canonical_1_12(X_char, cycles)  # [15,12,C,T]

        row_code, col_code = chara_map[ch]
        target_codes = {row_code, col_code}
        label_row = np.array([1 if (k in target_codes) else 0 for k in range(1,13)], dtype=np.int32)

        for start_rep in range(15 - group_size + 1):  
            X_block = X_ord[start_rep:start_rep+group_size]              
            y_block = np.stack([label_row.copy() for _ in range(group_size)], axis=0)  

            assert np.all(y_block.sum(axis=1) == 2), "target in some row are not 2"

            X_groups.append(X_block)  
            y_groups.append(y_block)  

    Xg = np.stack(X_groups, axis=0)  
    yg = np.stack(y_groups, axis=0)  
    return Xg, yg


def print_groups_for_char(yg, targets, char_idx=0, group_size=2, reps_per_char=15, max_groups=None):
    num_groups_per_char = reps_per_char - group_size + 1
    base = char_idx * num_groups_per_char
    end  = base + num_groups_per_char

    if max_groups is not None:
        end = min(end, base + max_groups)

    ch = targets[char_idx]
    print(f"\n=== Char #{char_idx}: {ch} | group_size={group_size} | groups={num_groups_per_char} ===")
    row_code, col_code = chara_map[ch]
    print(f"Expected 1-columns: {row_code} and {col_code} (1..12 indexing)")

    for gi, g in enumerate(range(base, end)):
        print(f"\nGroup {gi} (global idx {g}):")
        for r in range(group_size):
            row = yg[g, r]  
            one_cols = [i+1 for i, v in enumerate(row) if v == 1]  # 1-based index
            print(row, "  -> ones at", one_cols)


def reorder_trials_to_canonical_1_12(X_char_180, cycles_15x12):
    C, T = X_char_180.shape[1], X_char_180.shape[2]
    X_ord = np.empty((15, 12, C, T), dtype=X_char_180.dtype)
    for r in range(15):
        order = cycles_15x12[r]                 
        inv = {val: pos for pos, val in enumerate(order)}  
        for k in range(1, 13):                  
            pos_in_time = inv[k]
            trial_idx   = r * 12 + pos_in_time
            X_ord[r, k-1] = X_char_180[trial_idx]
    return X_ord
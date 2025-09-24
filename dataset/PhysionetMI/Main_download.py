import os
import sys
import pickle
import numpy as np
import mne
from scipy.signal import resample
from moabb.paradigms import MotorImagery
from moabb.datasets import PhysionetMI
from utility.data_utils import DataUtils

# Config
dataset_name = "PhysionetMI"
dataset_class = PhysionetMI
paradigm = MotorImagery()
desired_duration_sec = 3
sampling_rate = 160
re_sampling = True
preprocessing = False
re_sampling_rate = 240
target_len = desired_duration_sec * sampling_rate

# Path
RAW_DIR = "/data/MOABB/RAW"
if preprocessing and re_sampling:
    PROCESSED_DIR = f"/data/MOABB/Processed_filter_zscore/{dataset_name}"
elif re_sampling and not preprocessing:
    PROCESSED_DIR = f"/data/MOABB/Processed/{dataset_name}"
elif preprocessing and not re_sampling:
    PROCESSED_DIR = f"/data/EEGMS/Processed/filter_zscore/MOABB/{dataset_name}"
else:
    PROCESSED_DIR = f"/data/MOABB/Processed_RAW_Hz/{dataset_name}"
LOG_PATH = os.path.join(RAW_DIR, f"{dataset_name}.log")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.environ["MNE_DATA"] = RAW_DIR
assert os.path.exists(RAW_DIR), "MNE_DATA path missing"

sys.stdout = open(LOG_PATH, "w")
print(f"[LOG] MNE_DATA: {mne.get_config('MNE_DATA')}")

# data
dataset = dataset_class()
all_subjects = dataset.subject_list
print(f"[INFO] Dataset: {dataset_name}")
print(f"[INFO] Subjects: {len(all_subjects)} -> {all_subjects}")
print("----------------------------")
print(f"[INFO] Base fs: {sampling_rate} Hz")
print(f"[INFO] Duration: {desired_duration_sec} s")
print(f"[INFO] Resample: {re_sampling}")
print(f"[INFO] Resample fs: {re_sampling_rate} Hz" if re_sampling else "[INFO] Resample disabled")
print("----------------------------")

def collect_valid_subjects(ds, par, tgt_len):
    valid = []
    for s in ds.subject_list:
        try:
            X, y, _ = par.get_data(dataset=ds, subjects=[s])
            if isinstance(X, list):
                if min(tr.shape[1] for tr in X) < tgt_len:
                    continue
            else:
                if X.shape[2] < tgt_len:
                    continue
            valid.append(s)
        except Exception as e:
            print(f"[SKIP] Subject {s} failed: {e}")
    return valid

valid_subjects = collect_valid_subjects(dataset, paradigm, target_len)
n_total = len(valid_subjects)
n_train = int(n_total * 0.6)
n_val = int(n_total * 0.2)
n_test = n_total - n_train - n_val

train_subs = valid_subjects[:n_train]
val_subs = valid_subjects[n_train:n_train + n_val]
test_subs = valid_subjects[n_train + n_val:]

print(f"[INFO] Valid subjects: {n_total}")
print(f"[INFO] Train: {len(train_subs)}  Val: {len(val_subs)}  Test: {len(test_subs)}")
print(f"[SUBJECTS] Train: {train_subs}")
print(f"[SUBJECTS] Val:   {val_subs}")
print(f"[SUBJECTS] Test:  {test_subs}")

def load_data(subjects, tgt_len, do_resample, fs_new):
    all_X, all_y = [], []
    min_ch = None
    for s in subjects:
        try:
            print(f"[LOAD] Subject {s}")
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[s])
            if len(X) == 0:
                print(f"[WARN] Subject {s} empty")
                continue

            subj_min_ch = min(tr.shape[0] for tr in X) if isinstance(X, list) else X.shape[1]
            if isinstance(X, list):
                X = np.array([tr[:subj_min_ch, :tgt_len] for tr in X])
            else:
                if X.shape[2] < tgt_len:
                    print(f"[SKIP] Subject {s} too short: {X.shape[2]} < {tgt_len}")
                    continue
                X = X[:, :subj_min_ch, :tgt_len]

            if do_resample:
                new_len = int(fs_new * desired_duration_sec)
                X = resample(X, new_len, axis=2)

            min_ch = subj_min_ch if min_ch is None else min(min_ch, subj_min_ch)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"[ERROR] Subject {s} load error: {e}")

    try:
        print(f"[INFO] Align to {min_ch} channels")
        all_X = [x[:, :min_ch, :] for x in all_X]
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        return X, y
    except Exception as e:
        print(f"[FATAL] Concatenate error: {e}")
        return None, None

def safe_shape(tag, pair):
    try:
        X, y = pair
        print(f"{tag} X: {None if X is None else X.shape}  y: {None if y is None else y.shape}")
    except Exception as e:
        print(f"{tag} print error: {e}")

print("- load train")
X_train, Y_train = load_data(train_subs, target_len, re_sampling, re_sampling_rate)
print("- load val")
X_val,   Y_val   = load_data(val_subs,   target_len, re_sampling, re_sampling_rate)
print("- load test")
X_test,  Y_test  = load_data(test_subs,  target_len, re_sampling, re_sampling_rate)

if preprocessing:
    print("[INFO] Preprocess: bandpass and zscore")
    fs_filt = re_sampling_rate if re_sampling else sampling_rate
    X_train = DataUtils.chebyBandpassFilter(X_train, [0.2, 0.5, 40, 48], 40, 1, fs=fs_filt)
    X_val   = DataUtils.chebyBandpassFilter(X_val,   [0.2, 0.5, 40, 48], 40, 1, fs=fs_filt)
    X_test  = DataUtils.chebyBandpassFilter(X_test,  [0.2, 0.5, 40, 48], 40, 1, fs=fs_filt)
    def zscore(a): return (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-6)
    X_train = zscore(X_train)
    X_val   = zscore(X_val)
    X_test  = zscore(X_test)

if any(d is None for d in [X_train, X_val, X_test]):
    print("[ERROR] Split load failed. Skip save.")
else:
    save_path = os.path.join(PROCESSED_DIR, f"{dataset_name}.npy")
    save_dict = {
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "X_test":  X_test,  "Y_test":  Y_test,
    }
    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"[OK] Saved: {save_path}")

safe_shape("Train", (X_train, Y_train))
safe_shape("Val",   (X_val,   Y_val))
safe_shape("Test",  (X_test,  Y_test))

sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Log saved to: {LOG_PATH}")
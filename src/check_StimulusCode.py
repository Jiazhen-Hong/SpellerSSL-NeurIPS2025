import numpy as np

def verify_stimulus_code_order(stimulus_code, target_chars):
    """
    Verify if the 12-flash sequence is consistent across all repetitions.
    Supports both III2a [char, 1, time] and II2b [1, time] format.
    """
    stim_code = stimulus_code

    if stim_code.ndim == 2 and stim_code.shape[0] == 1:
        stim_code = stim_code[0]  

    if stim_code.ndim == 1:
        print("Detected flat StimulusCode — assuming II2b format")
        flash_locs = np.where(stim_code != 0)[0]

        if len(flash_locs) % 12 != 0:
            raise ValueError("Flash count not divisible by 12")

        rep_count = len(flash_locs) // 12
        flash_seq = stim_code[flash_locs].reshape(rep_count, 12)

        base = flash_seq[0]
        for r in range(1, rep_count):
            if not np.array_equal(base, flash_seq[r]):
                print(f"[✗] StimulusCode mismatch at repetition {r}")
                print("  Base:", base)
                print("  This:", flash_seq[r])
                return
        print("StimulusCode is consistent across all repetitions")
        return

    if stim_code.ndim == 3:
        num_chars = stim_code.shape[0]
        for i in range(num_chars):
            code = stim_code[i][0]
            flash_locs = np.where(code != 0)[0]

            if len(flash_locs) % 12 != 0:
                print(f"[!] Character {i} ({target_chars[i]}): flash count not divisible by 12")
                continue

            reps = len(flash_locs) // 12
            flash_groups = code[flash_locs].reshape(reps, 12)

            for r in range(1, reps):
                if not np.array_equal(flash_groups[r], flash_groups[0]):
                    print(f"[✗] StimulusCode mismatch for character {i} ({target_chars[i]}) at repetition {r}")
                    print("  Base:", flash_groups[0])
                    print("  This:", flash_groups[r])
                    break
            else:
                print(f"Character {i} ({target_chars[i]}): StimulusCode consistent")
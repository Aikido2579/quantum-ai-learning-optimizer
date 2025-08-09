import os
import numpy as np
import pandas as pd

# Dummy-safe imports
try:
    import mne
except ImportError:
    mne = None
try:
    import nibabel as nib
    from nilearn import image
except ImportError:
    nib = None
    image = None

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_mri(input_file="data/raw/sample_mri.nii.gz", output_file="data/processed/mri_features.csv"):
    if nib is None or image is None:
        print("[MRI] nibabel/nilearn not installed. Generating dummy features.")
        features = pd.DataFrame({"ROI1": [0.1], "ROI2": [0.2]})
    elif not os.path.exists(input_file):
        print(f"[MRI] No MRI file found at {input_file}. Generating dummy features.")
        features = pd.DataFrame({"ROI1": [0.1], "ROI2": [0.2]})
    else:
        img = nib.load(input_file)
        img_resampled = image.resample_img(img, target_affine=np.eye(3) * 2)
        data = img_resampled.get_fdata()
        features = pd.DataFrame({"mean_intensity": [np.mean(data)], "std_intensity": [np.std(data)]})
    features.to_csv(output_file, index=False)
    print(f"[MRI] Features saved to {output_file}")

def preprocess_eeg(input_file="data/raw/sample_eeg.fif", output_file="data/processed/eeg_features.csv"):
    if mne is None:
        print("[EEG] mne not installed. Generating dummy features.")
        features = pd.DataFrame({"alpha_power": [0.5], "beta_power": [0.3]})
    elif not os.path.exists(input_file):
        print(f"[EEG] No EEG file found at {input_file}. Generating dummy features.")
        features = pd.DataFrame({"alpha_power": [0.5], "beta_power": [0.3]})
    else:
        raw = mne.io.read_raw_fif(input_file, preload=True)
        raw.filter(1., 40.)
        psd, freqs = mne.time_frequency.psd_welch(raw)
        features = pd.DataFrame({"alpha_power": [np.mean(psd[:, (freqs >= 8) & (freqs <= 12)])],
                                 "beta_power": [np.mean(psd[:, (freqs >= 13) & (freqs <= 30)])]})
    features.to_csv(output_file, index=False)
    print(f"[EEG] Features saved to {output_file}")

def preprocess_learning_logs(input_file="data/raw/sample_logs.csv", output_file="data/processed/log_features.csv"):
    if not os.path.exists(input_file):
        print(f"[Logs] No log file found at {input_file}. Generating dummy features.")
        features = pd.DataFrame({"avg_correct": [0.75], "avg_time": [12.5]})
    else:
        df = pd.read_csv(input_file)
        features = pd.DataFrame({
            "avg_correct": [df["correct"].mean()],
            "avg_time": [df["time"].mean()]
        })
    features.to_csv(output_file, index=False)
    print(f"[Logs] Features saved to {output_file}")

if __name__ == "__main__":
    preprocess_mri()
    preprocess_eeg()
    preprocess_learning_logs()

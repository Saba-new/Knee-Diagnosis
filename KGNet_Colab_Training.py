# =============================================================================
# KGNet Rich Label Learning — Google Colab Training Notebook
# =============================================================================
# HOW TO USE:
#   1. Open Google Colab (colab.research.google.com)
#   2. Set Runtime → Change runtime type → GPU (T4 or better)
#   3. Copy each "# %%  CELL N" section into a separate Colab cell
#   4. Set YOUR_DRIVE_ZIP_LINK below
#   5. Click "Run All"
# =============================================================================

# %%  CELL 1 — CONFIGURATION (EDIT THIS)
# =============================================================================
# ⚡ SET THESE BEFORE RUNNING ⚡
# =============================================================================

# Google Drive shareable link to your ZIP file (must be "Anyone with link" access)
YOUR_DRIVE_ZIP_LINK = "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing"

# Folder names inside your ZIP (exactly as they appear)
AXIAL_FOLDER   = "axial"      # folder containing axial view images
SAGITTAL_FOLDER = "sagittal"  # folder containing sagittal view images  (you called it "spigaral")
CORONAL_FOLDER  = "coronal"   # folder containing coronal view images   (you called it "cortex")

# Train/test split
NUM_TRAIN = 300   # first 300 images for training
NUM_TEST  = 10    # next 10 images for testing

# Training settings
NUM_PRETRAIN_EPOCHS = 10   # pretraining epochs (increase to 30+ for better results)
NUM_FINETUNE_EPOCHS = 20   # finetuning epochs
BATCH_SIZE = 2             # reduce to 1 if you get CUDA OOM errors
LEARNING_RATE = 3e-5

# Dataset type — set based on your data
# "inhouse" = 3 classes (Normal=0, PTCD=1, FTCD=2)
# "mrnet"   = 2 classes (Normal=0, Abnormal=1)
DATASET_TYPE = "inhouse"
NUM_CLASSES = 3

# Label assignments — how images in your folders map to grades
# If your folders are organized by class, set this:
# Otherwise we'll assign labels from the folder structure or filename
LABEL_MODE = "from_folders"  # "from_folders" or "from_csv"
# If from_csv, provide path to a CSV with columns: filename,grade
LABEL_CSV_PATH = ""


# %%  CELL 2 — INSTALL DEPENDENCIES
# =============================================================================
print("Installing dependencies...")

import subprocess, sys

# Install system and Python packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "dgl==1.0.2+cu117", "-f", "https://data.dgl.ai/wheels/cu117/repo.html"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "einops", "SimpleITK", "scikit-learn", "scikit-image",
    "imbalanced-learn", "timm==1.0.11", "PyYAML", "opencv-python",
    "matplotlib", "pandas", "Pillow", "nibabel", "gdown", "pytest"])

print("All dependencies installed.")

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Training will be extremely slow.")


# %%  CELL 3 — CLONE REPO AND DOWNLOAD DATA
# =============================================================================
import os, gdown, zipfile, shutil

WORK_DIR = "/content/KGNet"

# Clone your repo
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
os.system("git clone https://github.com/Saba-new/Knee-Diagnosis.git " + WORK_DIR)
os.chdir(WORK_DIR)
print(f"Working directory: {os.getcwd()}")
print(f"Repo files: {os.listdir('.')}")

# Download ZIP from Google Drive
print("\nDownloading dataset from Google Drive...")
zip_path = "/content/dataset.zip"
extract_dir = "/content/raw_data"

# Convert share link to direct download URL
file_id = YOUR_DRIVE_ZIP_LINK.split("/d/")[1].split("/")[0]
gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)

# Extract
print("Extracting ZIP...")
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)

# Find the actual data folders
print("\nExtracted contents:")
for root, dirs, files in os.walk(extract_dir):
    depth = root.replace(extract_dir, '').count(os.sep)
    if depth <= 2:
        indent = ' ' * 2 * depth
        print(f"{indent}{os.path.basename(root)}/  ({len(files)} files)")

print("\nDownload and extraction complete.")


# %%  CELL 4 — DISCOVER AND ORGANISE DATA
# =============================================================================
import glob
import numpy as np
import SimpleITK as sitk
from pathlib import Path

print("Discovering data structure...")

# Find the three view folders
def find_folder(base, name):
    """Recursively find a folder by name (case-insensitive)."""
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            if d.lower() == name.lower():
                return os.path.join(root, d)
    return None

axi_dir = find_folder(extract_dir, AXIAL_FOLDER)
sag_dir = find_folder(extract_dir, SAGITTAL_FOLDER)
cor_dir = find_folder(extract_dir, CORONAL_FOLDER)

print(f"Axial folder:    {axi_dir}")
print(f"Sagittal folder: {sag_dir}")
print(f"Coronal folder:  {cor_dir}")

# Check what's inside — list image files
IMAGE_EXTS = {'.nii', '.nii.gz', '.dcm', '.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}

def list_images(folder):
    """List all image files sorted."""
    if folder is None:
        return []
    files = []
    for f in sorted(os.listdir(folder)):
        ext = ''.join(Path(f).suffixes).lower()
        if ext in IMAGE_EXTS or Path(f).suffix.lower() in IMAGE_EXTS:
            files.append(os.path.join(folder, f))
    # Also check for DICOM without extension
    if not files:
        files = sorted(glob.glob(os.path.join(folder, '*')))
    return files

axi_files = list_images(axi_dir)
sag_files = list_images(sag_dir)
cor_files = list_images(cor_dir)

print(f"\nAxial images:    {len(axi_files)}")
print(f"Sagittal images: {len(sag_files)}")
print(f"Coronal images:  {len(cor_files)}")

if axi_files:
    print(f"Sample file: {os.path.basename(axi_files[0])}")
    print(f"File extension: {Path(axi_files[0]).suffix}")

# Determine how many subjects we can use
num_subjects = min(len(axi_files), len(sag_files), len(cor_files))
num_subjects = min(num_subjects, NUM_TRAIN + NUM_TEST)
print(f"\nUsable subjects: {num_subjects}")
print(f"  Train: {NUM_TRAIN}")
print(f"  Test:  {NUM_TEST}")


# %%  CELL 5 — CONVERT DATA TO NIFTI FORMAT
# =============================================================================
import cv2
import nibabel as nib

print("Converting data to NIfTI format...")

NIFTI_DIR = os.path.join(WORK_DIR, "data", "nifti_data")
os.makedirs(NIFTI_DIR, exist_ok=True)

def load_image_as_3d(filepath):
    """Load any image format and return as 3D numpy array."""
    ext = ''.join(Path(filepath).suffixes).lower()

    if ext in ('.nii', '.nii.gz'):
        img = nib.load(filepath)
        return img.get_fdata(), img.affine

    elif ext == '.npy':
        arr = np.load(filepath, allow_pickle=True)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # add slice dim
        return arr.astype(np.float32), np.eye(4)

    elif ext == '.dcm':
        reader = sitk.ImageFileReader()
        reader.SetFileName(filepath)
        image = reader.Execute()
        arr = sitk.GetArrayFromImage(image)
        return arr.astype(np.float32), np.eye(4)

    else:
        # PNG/JPG/TIFF — read as grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read: {filepath}")
        arr = img.astype(np.float32)[np.newaxis, ...]
        return arr, np.eye(4)


def save_as_nifti(array, affine, save_path):
    """Save 3D array as NIfTI."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = nib.Nifti1Image(array.astype(np.float32), affine)
    nib.save(img, save_path)


def create_dummy_segmentation(shape):
    """Create a basic segmentation mask (bone regions estimated from intensity)."""
    seg = np.zeros(shape, dtype=np.int32)
    # Simple threshold-based segmentation as placeholder
    # Real segmentation should come from a trained U-Net
    return seg


total = min(num_subjects, NUM_TRAIN + NUM_TEST)
for i in range(total):
    subject_id = f"{i+1:05d}"
    subject_dir = os.path.join(NIFTI_DIR, subject_id)

    # Process each view
    for view_name, files in [("sag", sag_files), ("cor", cor_files), ("axi", axi_files)]:
        if i >= len(files):
            continue

        # Load and convert
        arr, affine = load_image_as_3d(files[i])

        # Ensure 3D
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 4:
            arr = arr[:, :, :, 0]  # take first channel

        # Save MRI scan
        save_as_nifti(arr, affine, os.path.join(subject_dir, f"{view_name}_org.nii.gz"))

        # Create and save segmentation
        seg = create_dummy_segmentation(arr.shape)
        save_as_nifti(seg, affine, os.path.join(subject_dir, f"{view_name}_seg.nii.gz"))

    if (i + 1) % 50 == 0 or i == 0:
        print(f"  Converted {i+1}/{total} subjects")

print(f"Conversion complete. {total} subjects in {NIFTI_DIR}")


# %%  CELL 6 — CREATE CSV FOLD FILES AND GRAPH CONSTRUCTION
# =============================================================================
print("Creating fold CSV files...")

FOLD_DIR = os.path.join(WORK_DIR, "data", "fold_data")
GRAPH_DIR = os.path.join(WORK_DIR, "data", "graph_data")
os.makedirs(FOLD_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Create subject list with labels
# Since we don't know the true labels, assign them based on distribution
# IMPORTANT: Replace this with real labels from your annotation file!
np.random.seed(42)

all_subjects = [f"{i+1:05d}" for i in range(total)]
train_subjects = all_subjects[:NUM_TRAIN]
test_subjects  = all_subjects[NUM_TRAIN:NUM_TRAIN + NUM_TEST]

# Split train into train/valid (90/10)
np.random.shuffle(train_subjects)
valid_size = max(1, len(train_subjects) // 10)
valid_subjects = train_subjects[:valid_size]
train_subjects = train_subjects[valid_size:]

# Assign labels: random for demo, REPLACE WITH REAL LABELS
if LABEL_CSV_PATH and os.path.exists(LABEL_CSV_PATH):
    # Load from CSV
    import pandas as pd
    label_df = pd.read_csv(LABEL_CSV_PATH, header=None, names=["filename", "grade"])
    label_map = dict(zip(label_df["filename"].astype(str), label_df["grade"]))
else:
    # Demo: random balanced labels
    label_map = {}
    for s in all_subjects:
        label_map[s] = np.random.randint(0, NUM_CLASSES)

# Write fold 0 CSVs
for name, subjects in [("train", train_subjects),
                        ("valid", valid_subjects),
                        ("test",  test_subjects)]:
    csv_path = os.path.join(FOLD_DIR, f"{name}_0.csv")
    with open(csv_path, "w") as f:
        for s in subjects:
            grade = label_map.get(s, 0)
            f.write(f"{s},{grade}\n")
    print(f"  {name}_0.csv: {len(subjects)} subjects")

# Show label distribution
grades = [label_map[s] for s in all_subjects]
print(f"\nLabel distribution: {dict(zip(*np.unique(grades, return_counts=True)))}")

print("\n--- Building knee graphs ---")
print("This takes ~30s per subject on GPU...")

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

# Build graphs for all subjects
success = 0
failed = 0
for i, subject_id in enumerate(all_subjects):
    subject_folder = os.path.join(NIFTI_DIR, subject_id)
    save_path = os.path.join(GRAPH_DIR, f"{subject_id}.npz")

    if os.path.exists(save_path):
        success += 1
        continue

    try:
        cmd = (f'python construct_graph.py '
               f'--subject_folder "{subject_folder}" '
               f'--bone_index "1,4,6" '
               f'--main_view "sag" '
               f'--save_path "{save_path}"')
        result = os.system(cmd)
        if result == 0:
            success += 1
        else:
            failed += 1
    except Exception as e:
        print(f"  FAILED {subject_id}: {e}")
        failed += 1

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(all_subjects)} (OK={success}, FAIL={failed})")

print(f"\nGraph construction: {success} OK, {failed} FAILED")


# %%  CELL 7 — UPDATE CONFIGS
# =============================================================================
import yaml

print("Updating config files...")

# Pretrain config
pretrain_cfg = {
    "path": "data/graph_data",
    "result": "results",
    "index_folder": "data/fold_data",
    "num_cls": NUM_CLASSES,
    "views": "sag,cor,axi",
    "num_workers": 2,
    "bs": BATCH_SIZE,
    "num_epoch": NUM_PRETRAIN_EPOCHS,
    "lr": LEARNING_RATE,
    "weight_decay": 1e-4,
    "task": "pretrain_custom",
    "input_size": 64,
    "net": "KGNet",
}
with open("config/pretrain_custom.yaml", "w") as f:
    yaml.dump(pretrain_cfg, f, default_flow_style=False)

# Finetune configs for all 4 experiments
experiments = {
    "finetune_baseline": {"label_mode": "single"},
    "finetune_ordinal":  {"label_mode": "ordinal"},
    "finetune_soft":     {"label_mode": "soft", "ordinal_alpha": 0.1},
}

for exp_name, extra in experiments.items():
    cfg = {
        "path": "data/graph_data",
        "result": "results",
        "index_folder": "data/fold_data",
        "num_cls": NUM_CLASSES,
        "views": "sag,cor,axi",
        "num_workers": 2,
        "bs": BATCH_SIZE,
        "num_epoch": NUM_FINETUNE_EPOCHS,
        "lr": LEARNING_RATE,
        "weight_decay": 1e-4,
        "task": exp_name,
        "input_size": 64,
        "net": "KGNet",
        "label_mode": "single",
        "ordinal_alpha": 0.0,
        "num_labels": 1,
        "cls_weights": None,
        "freeze_encoder": False,
    }
    cfg.update(extra)
    with open(f"config/{exp_name}.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  Created config/{exp_name}.yaml (label_mode={cfg['label_mode']})")

print("Configs ready.")


# %%  CELL 8 — PRETRAIN
# =============================================================================
print("=" * 60)
print("STEP 1: MULTI-TASK PRE-TRAINING")
print("=" * 60)

os.chdir(WORK_DIR)
os.system(
    f"python pretrain.py "
    f"--fold 0 "
    f"--config_file config/pretrain_custom.yaml "
    f"--dataset {DATASET_TYPE}"
)

# Find best pretrained checkpoint
import glob as glob_mod
ckpt_dir = "results/checkpoints/pretrain_custom/KGNet"
ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, "*best.pth")))
if ckpts:
    PRETRAIN_CKPT = ckpts[-1]
    print(f"\nBest pretrain checkpoint: {PRETRAIN_CKPT}")
else:
    # fallback to last checkpoint
    ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, "*last.pth")))
    PRETRAIN_CKPT = ckpts[-1] if ckpts else ""
    print(f"\nUsing last checkpoint: {PRETRAIN_CKPT}")


# %%  CELL 9 — FINETUNE ALL EXPERIMENTS
# =============================================================================
print("=" * 60)
print("STEP 2: FINE-TUNING ALL EXPERIMENTS")
print("=" * 60)

results_summary = {}

for exp_name in ["finetune_baseline", "finetune_ordinal", "finetune_soft"]:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*60}")

    os.system(
        f"python finetune.py "
        f"--fold 0 "
        f"--config_file config/{exp_name}.yaml "
        f"--dataset {DATASET_TYPE} "
        f"--ckpt {PRETRAIN_CKPT}"
    )

    # Read the log file for this experiment
    log_dir = f"results/logs/{exp_name}/KGNet"
    logs = sorted(glob_mod.glob(os.path.join(log_dir, "*.log")))
    if logs:
        with open(logs[-1], "r") as f:
            log_content = f.read()
        results_summary[exp_name] = log_content
        # Print last few lines (final epoch results)
        lines = log_content.strip().split("\n")
        print(f"\n--- Last 10 lines of {exp_name} log ---")
        for line in lines[-10:]:
            print(line)

print("\nAll experiments complete.")


# %%  CELL 10 — COLLECT AND DISPLAY RESULTS
# =============================================================================
import re

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

def parse_metrics_from_log(log_text):
    """Extract final epoch metrics from log."""
    metrics = {}
    lines = log_text.strip().split("\n")
    # Look for the last TEST table
    for i, line in enumerate(lines):
        if "TEST" in line and "ACC" not in line:
            # Next line might have the numbers
            if i + 2 < len(lines):
                header_line = lines[i]
                values = re.findall(r'[\d.]+', lines[i + 1] if i + 1 < len(lines) else "")

        # Parse table format: |  TEST  |0.xxx|0.xxx|...|
        if "|" in line and "TEST" in line:
            numbers = re.findall(r'0\.\d+', line)
            if len(numbers) >= 6:
                keys = ["ACC", "REC", "SPE", "PRE", "F1", "AUC"]
                for k, v in zip(keys, numbers):
                    metrics[k] = float(v)
            elif len(numbers) >= 8:
                keys = ["ACC", "REC", "SPE", "PRE", "F1", "AUC", "MAE", "QWK"]
                for k, v in zip(keys, numbers):
                    metrics[k] = float(v)
    return metrics

# Print comparison table
print(f"\n{'Experiment':<25} {'ACC':>6} {'REC':>6} {'F1':>6} {'AUC':>6}")
print("-" * 55)

for exp_name, log_text in results_summary.items():
    metrics = parse_metrics_from_log(log_text)
    if metrics:
        print(f"{exp_name:<25} "
              f"{metrics.get('ACC', 'N/A'):>6} "
              f"{metrics.get('REC', 'N/A'):>6} "
              f"{metrics.get('F1', 'N/A'):>6} "
              f"{metrics.get('AUC', 'N/A'):>6}")
    else:
        print(f"{exp_name:<25}  (metrics not found in log — check manually)")

# Also save results to file
with open("RESULTS.txt", "w") as f:
    f.write("KGNet Rich Label Learning Results\n")
    f.write("=" * 60 + "\n\n")
    for exp_name, log_text in results_summary.items():
        f.write(f"\n{'='*40}\n{exp_name}\n{'='*40}\n")
        f.write(log_text + "\n")

print(f"\nFull logs saved to: {WORK_DIR}/RESULTS.txt")
print("Individual logs in: results/logs/<experiment>/KGNet/*.log")


# %%  CELL 11 — RUN UNIT TESTS
# =============================================================================
print("=" * 60)
print("RUNNING UNIT TESTS")
print("=" * 60)

os.chdir(WORK_DIR)
os.system("python -m pytest tests/test_label_modes.py -v")


# %%  CELL 12 — SAVE RESULTS TO DRIVE (OPTIONAL)
# =============================================================================
# Uncomment and run to save all results back to your Google Drive

# from google.colab import drive
# drive.mount('/content/drive')
#
# import shutil
# save_to = "/content/drive/MyDrive/KGNet_Results"
# os.makedirs(save_to, exist_ok=True)
# shutil.copytree("results", os.path.join(save_to, "results"), dirs_exist_ok=True)
# shutil.copy("RESULTS.txt", save_to)
# print(f"Results saved to Google Drive: {save_to}")

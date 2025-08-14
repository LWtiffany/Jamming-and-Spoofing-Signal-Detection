import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import glob
import time
from sklearn.model_selection import train_test_split
import torch

# ========= constants =========
SAMPLE_RATE       = 25_000_000          # 25 Msps
BYTES_PER_SAMPLE  = 4                   # int16 I + int16 Q
CLIP_RANGE_DB     = 80                  # dynamic range (80 dB)

# ========= read IQ =========
def read_texbat(path: str,
                start_sec: float = 0,
                duration_sec: float = None,
                mmap: bool = False) -> np.ndarray:
    path       = Path(path)
    start_samp = int(start_sec * SAMPLE_RATE)
    byte_off   = start_samp * BYTES_PER_SAMPLE

    if duration_sec is None:
        count = -1                       # read to EOF
    else:
        n_samp = int(duration_sec * SAMPLE_RATE)
        count  = n_samp * 2              # I and Q

    if mmap:
        data = np.memmap(path, dtype='<i2', mode='r',
                         offset=byte_off, shape=(count if count>0 else None))
    else:
        with path.open('rb') as f:
            f.seek(byte_off)
            data = np.fromfile(f, dtype='<i2', count=count)

    # normalize to [-1, 1] to prevent dB calculation overflow
    iq = (data[0::2] + 1j * data[1::2]).astype(np.complex64) / 32768.0
    return iq

# ========= simple 2D average pooling =========
def pool_spec(spec: np.ndarray,
              pool_h: int = 4,
              pool_w: int = 128) -> np.ndarray:
    h = spec.shape[0] - (spec.shape[0] % pool_h)
    w = spec.shape[1] - (spec.shape[1] % pool_w)
    spec = spec[:h, :w]
    spec = spec.reshape(h // pool_h, pool_h, w).mean(1)
    spec = spec.reshape(spec.shape[0], w // pool_w, pool_w).mean(2)
    return spec

# ========= main function: bin -> time-frequency image =========
def dump_images(bin_path, out_dir, win_sec=10, hop_sec=2):
    from matplotlib import pyplot as plt
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"start processing: {bin_path}")
    start_time = time.time()

    total_sec = Path(bin_path).stat().st_size / (BYTES_PER_SAMPLE * SAMPLE_RATE)
    start, idx = 0.0, 0
    while start + win_sec <= total_sec:
        iq = read_texbat(bin_path, start_sec=start, duration_sec=win_sec)

        # downsample to 5 Msps
        iq = signal.decimate(iq, 5)
        fs = SAMPLE_RATE // 5

        f, t, Zxx = signal.stft(iq, fs=fs,
                                window='hann',
                                nperseg=1024,
                                noverlap=512,
                                return_onesided=False,
                                boundary=None)

        # only take ±2 MHz
        band = np.where((f >= -2e6) & (f <= 2e6))[0]
        spec = 20 * np.log10(np.abs(Zxx[band]) + 1e-12)   # add 1e-12 to prevent log(0)

        # automatic dynamic range: take 99th percentile as vmax, then subtract 80 dB
        vmax = np.percentile(spec, 99)
        vmin = vmax - CLIP_RANGE_DB
        spec = np.clip(spec, vmin, vmax)
        spec = (spec - vmin) / (vmax - vmin)              # map to [0,1]

        # after pooling, interpolate to 224×224, convenient for MobileNet
        spec = pool_spec(spec, pool_h=4, pool_w=128)
        spec = signal.resample(spec, 224, axis=0)         # frequency dimension
        spec = signal.resample(spec, 224, axis=1)         # time dimension

        plt.imsave(f'{out_dir}/{idx:06d}.png', spec, cmap='viridis')
        start += hop_sec
        idx   += 1
    elapsed_time = time.time() - start_time
    print(f"completed processing: {bin_path}, time: {elapsed_time:.2f}s, total {idx} images")

# ========= multiprocess processing =========
def process_single_file(args):
    bin_path, out_dir, win_sec, hop_sec = args
    dump_images(bin_path, out_dir, win_sec, hop_sec)

def dump_images_multiprocess(bin_files, base_out_dir, win_sec=10, hop_sec=2, n_processes=6):
    from multiprocessing import Pool
    tasks = []
    for i, bin_path in enumerate(bin_files):
        out_dir = f"{base_out_dir}/{Path(bin_path).stem}"
        tasks.append((bin_path, out_dir, win_sec, hop_sec))
    print(f"start multiprocessing {len(bin_files)} files, using {n_processes} processes")
    with Pool(processes=n_processes) as pool:
        pool.map(process_single_file, tasks)
    print("all files processed!")

# ========= dataset class =========
class SpoofingDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        if self.image_paths[idx].endswith('.png'):
            import cv2
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.float32) / 255.0
        else:
            image = np.load(self.image_paths[idx]).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def prepare_dataset(data_dir, test_size=0.2, val_size=0.1, window_size=5.0, stride=1.0):
    """
    Prepare dataset for spoofing signal detection with automatic reclassification.
    
    This function automatically reclassifies the first 100 seconds of data from spoofing 
    signal directories as real signals, which is realistic since spoofing attacks typically 
    don't start immediately.
    
    Args:
        data_dir (str): Root directory containing subdirectories for each class
        test_size (float): Proportion of dataset to include in test split (default: 0.2)
        val_size (float): Proportion of dataset to include in validation split (default: 0.1)
        window_size (float): Time window size in seconds for each sample (default: 1.0)
        stride (float): Time stride in seconds between consecutive samples (default: 1.0)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) - DataLoader objects for training,
               validation, and testing
    
    Note:
        - First 100 seconds of spoofing directories (ds1, ds2, ds3, ds4, ds7) are 
          automatically reclassified as real signals (class 0)
        - Number of reclassified images = 100 / stride
        - cleanDynamic directory samples remain as real signals (class 0)
    """
    image_paths = []
    labels = []
    
    # Calculate number of images corresponding to first 100 seconds
    # Using formula: N = floor((T - win_sec) / hop_sec) + 1
    # Where T=100s (total time), win_sec=window_size, hop_sec=stride
    first_100s_count = int((100.0 - window_size) / stride) + 1
    
    # Define class mapping
    class_mapping = {
        'cleanDynamic': 0,  # Real Signal
        'ds1': 1,          # RF Switch
        'ds2': 2,          # Over-powered Time-Push
        'ds3': 3,          # Matched-power Time-Push
        'ds4': 4,          # Matched-power Position-Push
        'ds7': 5           # Seamless Matched-power Time-Push
    }
    
    class_names = {
        0: 'Real Signal (cleanDynamic)',
        1: 'RF Switch (ds1)',
        2: 'Over-powered Time-Push (ds2)',
        3: 'Matched-power Time-Push (ds3)',
        4: 'Matched-power Position-Push (ds4)',
        5: 'Seamless Matched-power Time-Push (ds7)'
    }
    
    # Spoofing signal directories (exclude cleanDynamic)
    spoofing_dirs = ['ds1', 'ds2', 'ds3', 'ds4', 'ds7']
    
    class_counts = [0] * 6
    reclassified_counts = [0] * 6  # Track reclassified samples
    
    for dirname, original_class_label in class_mapping.items():
        dir_path = os.path.join(data_dir, dirname)
        if os.path.exists(dir_path):
            png_files = sorted(glob.glob(os.path.join(dir_path, '*.png')))
            
            if dirname in spoofing_dirs:
                # For spoofing signal directories
                for i, png_file in enumerate(png_files):
                    image_paths.append(png_file)
                    
                    # Reclassify first 100 seconds as real signal (class 0)
                    if i < first_100s_count:
                        labels.append(0)  # Real signal
                        class_counts[0] += 1
                        reclassified_counts[original_class_label] += 1
                    else:
                        labels.append(original_class_label)  # Original spoofing class
                        class_counts[original_class_label] += 1
                
                print(f"Load {class_names[original_class_label]}: {len(png_files)} samples")
                print(f"  - First {min(first_100s_count, len(png_files))} samples reclassified as Real Signal")
                print(f"  - Remaining {max(0, len(png_files) - first_100s_count)} samples kept as {class_names[original_class_label]}")
            else:
                # For cleanDynamic directory, keep all as real signal
                image_paths.extend(png_files)
                labels.extend([original_class_label] * len(png_files))
                class_counts[original_class_label] += len(png_files)
                print(f"Load {class_names[original_class_label]}: {len(png_files)} samples")
        else:
            print(f"Warning: Directory {dirname} not found")
    
    print(f"\nData label statistics after reclassification:")
    print(f"Total {len(image_paths)} samples")
    for i, count in enumerate(class_counts):
        if count > 0:
            if i == 0:  # Real signal class
                original_clean = count - sum(reclassified_counts)
                reclassified_total = sum(reclassified_counts)
                print(f"Class {i} ({class_names[i]}): {count} samples")
                print(f"  - Original cleanDynamic: {original_clean} samples")
                print(f"  - Reclassified from spoofing (first 100s): {reclassified_total} samples")
            else:
                print(f"Class {i} ({class_names[i]}): {count} samples")
    
    print(f"\nReclassification summary:")
    total_reclassified = sum(reclassified_counts)
    print(f"Total samples reclassified from spoofing to real signal: {total_reclassified}")
    for i, reclass_count in enumerate(reclassified_counts):
        if reclass_count > 0:
            print(f"  - From {class_names[i]}: {reclass_count} samples")
    
    if len(image_paths) == 0:
        raise ValueError("No image files found, please check data directory path")
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    val_size_adj = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adj, random_state=42, stratify=y_temp
    )
    print(f"\ndata set splitting:")
    print(f"training set: {len(X_train)} samples")
    print(f"validation set: {len(X_val)} samples")
    print(f"test set: {len(X_test)} samples")
    train_dataset = SpoofingDataset(X_train, y_train)
    val_dataset = SpoofingDataset(X_val, y_val)
    test_dataset = SpoofingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader 
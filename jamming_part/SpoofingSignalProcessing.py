
import os
import numpy as np
import struct

def read_bin_iq(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    float_data = struct.unpack('f' * (len(data) // 4), data)
    complex_data = np.array(float_data[::2]) + 1j * np.array(float_data[1::2])
    return complex_data

def load_spoofing_ranges(range_file):
    spoof_ranges = []
    with open(range_file, 'r') as f:
        for line in f:
            start, end = map(int, line.strip().split(','))
            spoof_ranges.append((start, end))
    return spoof_ranges

def is_spoofed(center_idx, spoof_ranges):
    for start, end in spoof_ranges:
        if start <= center_idx <= end:
            return True
    return False

def chunk_and_label(complex_data, spoof_ranges, output_dir, window_size=272, step_size=32, prefix='chunk'):
    os.makedirs(output_dir, exist_ok=True)
    label_lines = []
    complex_data = np.stack([np.real(complex_data), np.imag(complex_data)], axis=-1)

    for i in range(0, len(complex_data) - window_size, step_size):
        chunk = complex_data[i:i+window_size]
        center = i + window_size // 2
        label = 1 if is_spoofed(center, spoof_ranges) else 0
        file_name = f"{prefix}_{i}.npy"
        np.save(os.path.join(output_dir, file_name), chunk)
        label_lines.append(f"{file_name},{label}\n")

    return label_lines

def process_bin_with_ranges(bin_path, range_path, output_dir, prefix):
    complex_data = read_bin_iq(bin_path)
    spoof_ranges = load_spoofing_ranges(range_path)
    return chunk_and_label(complex_data, spoof_ranges, output_dir, prefix=prefix)

if __name__ == "__main__":
    bin_path = "D:/TEXBAT/DS7.bin"
    range_path = "D:/TEXBAT/DS7_spoofing_ranges.txt"
    output_dir = "D:/Dataset_DS7/chunks/"
    prefix = "DS7"

    label_lines = process_bin_with_ranges(bin_path, range_path, output_dir, prefix=prefix)
    with open(os.path.join(output_dir, f"{prefix}_labels.txt"), 'w') as f:
        f.writelines(label_lines)

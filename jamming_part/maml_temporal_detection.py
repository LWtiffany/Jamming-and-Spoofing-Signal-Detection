
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import struct

def read_bin_iq(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    float_data = struct.unpack('f' * (len(data) // 4), data)
    complex_data = np.array(float_data[::2]) + 1j * np.array(float_data[1::2])
    return complex_data

def load_spoofing_ranges(range_file):
    spoof_ranges = []
    if not os.path.exists(range_file):
        return spoof_ranges  # 返回空表示全是正常
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

def process_bin_to_chunks(dataset_name, bin_path, range_path, root='D:/Dataset_DS7/chunks/',
                          window_size=272, step_size=32):
    os.makedirs(root, exist_ok=True)
    spoof_ranges = load_spoofing_ranges(range_path)
    data = []
    labels = []

    complex_data = read_bin_iq(bin_path)
    complex_data = np.stack([np.real(complex_data), np.imag(complex_data)], axis=-1)

    label_lines = []
    for i in range(0, len(complex_data) - window_size, step_size):
        chunk = complex_data[i:i+window_size]
        center = i + window_size // 2
        label = 1 if is_spoofed(center, spoof_ranges) else 0

        file_name = f"{dataset_name}_{i}.npy"
        np.save(os.path.join(root, file_name), chunk)
        label_lines.append(f"{file_name},{label}\n")

    with open(os.path.join(root, f"{dataset_name}_labels.txt"), 'w') as f:
        f.writelines(label_lines)

def load_dataset(dataset_name, root='D:/Dataset_DS7/chunks/', window_size=256, pred_len=16):
    label_path = os.path.join(root, f"{dataset_name}_labels.txt")
    bin_path = f"D:/TEXBAT/{dataset_name}.bin"
    range_path = f"D:/TEXBAT/{dataset_name}_spoofing_ranges.txt"

    # 如果标签文件不存在，但 bin 存在，自动预处理生成
    if not os.path.exists(label_path) and os.path.exists(bin_path):
        print(f"[INFO] Processing raw bin for {dataset_name}...")
        process_bin_to_chunks(dataset_name, bin_path, range_path, root)

    data = []
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Cannot find label file: {label_path}")

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name, lbl = line.strip().split(',')
            arr = np.load(os.path.join(root, name)).astype(np.float32)
            data.append((arr, int(lbl)))
    return data


# ======== Model Definition ========
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_len=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim * output_len)
        self.output_len = output_len
        self.input_dim = input_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.output_len, self.input_dim)

# ======== Sampling ========
def sample_task_batch(task_data, window_size=256, pred_len=16, batch_size=8):
    inputs, targets = [], []
    for _ in range(batch_size):
        x, label = task_data[np.random.randint(len(task_data))]
        if len(x) < window_size + pred_len:
            continue
        start = np.random.randint(0, len(x) - window_size - pred_len)
        seq_in = x[start:start+window_size]
        seq_out = x[start+window_size:start+window_size+pred_len]
        inputs.append(seq_in)
        targets.append(seq_out)
    return torch.tensor(inputs), torch.tensor(targets)

# ======== MAML training ========
def maml_train(model, tasks, epochs=3, inner_steps=1, lr_inner=0.01, lr_outer=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr_outer)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for task_name, task_data in tasks.items():
            model_inner = copy.deepcopy(model)
            model_inner.to(device)
            optimizer_inner = optim.SGD(model_inner.parameters(), lr=lr_inner)

            for _ in range(inner_steps):
                x, y = sample_task_batch(task_data)
                x, y = x.to(device), y.to(device)
                loss = criterion(model_inner(x), y)
                optimizer_inner.zero_grad()
                loss.backward()
                optimizer_inner.step()

            x_val, y_val = sample_task_batch(task_data)
            x_val, y_val = x_val.to(device), y_val.to(device)
            y_pred = model_inner(x_val)
            outer_loss = criterion(y_pred, y_val)

            optimizer.zero_grad()
            outer_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch}] Outer loss: {outer_loss.item():.4f}")

# ======== exception detection ========
def detect_anomaly(model, test_data, threshold=0.01):
    model.eval()
    device = next(model.parameters()).device
    errors = []
    labels = []
    anomaly_indices = []

    for idx, (x, label) in enumerate(test_data):
        if len(x) < 272: continue
        x_input = torch.tensor(x[:256]).unsqueeze(0).to(device)
        y_true = torch.tensor(x[256:256+16]).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(x_input)
            mse = torch.mean((y_pred - y_true) ** 2).item()
            errors.append(mse)
            labels.append(label)
            if mse > threshold:
                anomaly_indices.append(idx)

    return errors, labels, anomaly_indices

if __name__ == "__main__":
    train_tasks = {f"DS{i}": load_dataset(f"DS{i}") for i in [1, 2, 3, 4]}
    test_data = load_dataset("DS7")

    model = LSTMPredictor()
    model_path = "maml_lstm_model.pth"

    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training model...")
        maml_train(model, train_tasks, epochs=5)
        torch.save(model.state_dict(), model_path)

    errors, labels, anomaly_indices = detect_anomaly(model, test_data)
    preds = [1 if e > 0.01 else 0 for e in errors]

    print("Anomaly Indices (predicted):", anomaly_indices)

    auc = roc_auc_score(labels, errors)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    plt.plot(errors, label='Prediction Error')
    plt.axhline(y=0.01, color='r', linestyle='--', label='Threshold')
    plt.title("DS7 Prediction Error (Anomaly Score)")
    plt.legend()
    plt.show()

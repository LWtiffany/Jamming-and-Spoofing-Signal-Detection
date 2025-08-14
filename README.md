# 🛰️ Jamming and Spoofing Signal Detection System

## 📋 Project Overview

This project is a deep learning-based GPS signal processing system, specifically designed for detecting and identifying jamming and spoofing attacks in GPS signals. The system leverages advanced machine learning algorithms, including meta-learning (MAML) and convolutional neural networks (MobileNetV2), to achieve intelligent analysis and threat detection of GPS signals.

## ✨ Main Features

- 🎯 **Multimodal Detection**：Support jamming signal detection and spoofing signal detection
- 🧠 **Advanced Algorithms**：Integrate MAML meta-learning and MobileNetV2 deep learning models
- 📊 **Complete Pipeline**：Include data preprocessing, model training, evaluation, and visualization
- ⚡ **High Performance**：Support CUDA acceleration, optimized training strategies
- 📈 **Visualization**：Provide training history, confusion matrix, and probability distribution graphs

## 🏗️ Project Structure

```
final code/
├── 📁 jamming_part/              # Jamming signal detection module
│   ├── 🔧 dataTransfer.py        # Data transfer and preprocessing
│   ├── 🤖 maml_temporal_detection.py  # MAML temporal detection algorithm
│   ├── 📊 SpoofingSignalProcessing.py # Signal processing tool
│   └── 📓 signal_classify_model*.ipynb # Signal classification model notebook
├── 📁 spoofing_part/             # Spoofing signal detection module (single version)
│   ├── 🔧 data_processing.py     # Data preprocessing
│   ├── 🤖 model.py              # MobileNetV2 model definition
│   ├── 🚀 spoofing_signal_detection.py # Main training script
│   ├── 🛠️ utils.py              # Utility functions and evaluation
│   ├── 📓 GPS_Spoofing_Detection_VAE.ipynb # VAE detection notebook
│   └── 📁 snapshot/             # Model and result saving
├── 📁 spoofing_part_multi/       # Spoofing signal detection module (multi-version)
│   └── [Same structure as spoofing_part]
├── 🌍 environment.yml           # Conda environment configuration
└── 📖 README.md                # Project documentation
```

## 🔧 Environment Configuration

### System Requirements
- Python 3.8+
- CUDA 11.8+ (Optional, for GPU acceleration)
- Recommended 8GB+ GPU memory

### Quick Installation

1. **Clone the project**
```bash
git clone <repository-url>
cd final_code
```

2. **Create a Conda environment**
```bash
conda env create -f environment.yml
conda activate signal-classifier
```

3. **Verify the installation**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 🚀 Quick Start

### 1. Jamming Signal Detection (MAML Method)

```bash
cd jamming_part
python maml_temporal_detection.py
```

**Main features:**
- Meta-learning algorithm based on MAML
- Time-series signal analysis
- Adaptive learning ability

### 2. Spoofing Signal Detection (MobileNetV2 Method)

```bash
cd spoofing_part
python spoofing_signal_detection.py
```

**Configuration parameters:**
```python
# Training configuration
NUM_EPOCHS = 30           # Number of training epochs
LEARNING_RATE = 0.00007   # Learning rate
NUM_CLASSES = 2           # Number of classes (real/spoof)
FREEZE_STRATEGY = 'backbone'  # Freezing strategy
```

### 3. Multi-version Spoofing Signal Detection

```bash
cd spoofing_part_multi
python spoofing_signal_detection.py
```

## 🧠 Model Architecture

### MAML Temporal Detector
- **Algorithm**：Model-Agnostic Meta-Learning
- **Features**：Fast adaptation to new scenarios
- **Input**：IQ complex signal sequence
- **Output**：Jamming detection result

### MobileNetV2 Classifier
- **Architecture**：Lightweight convolutional neural network
- **Features**：Efficient mobile-end inference
- **Input**：Spectrogram features
- **Output**：Real/spoof signal probability

## 📊 Performance Metrics

According to the latest test results:

| Metric | Value |
|------|------|
| 🎯 Overall accuracy | **95.6%** |
| 📈 Precision | **94.8%** |
| 📊 Recall | **96.2%** |
| 🔍 F1-Score | **95.5%** |

### Confusion Matrix
- Real signal detection rate: 97.1%
- Spoof signal detection rate: 94.3%
- False alarm rate: 2.9%

## 🔬 Data Processing Flow

### 1. Original Data Format
- **Input**：`.bin` format IQ signal file
- **Sampling rate**：25 MHz
- **Data type**：int16 (I/Q interleaved storage)

### 2. Preprocessing Steps
```python
# Read IQ data
iq_data = read_texbat(file_path)

# Compute spectrogram
spectrogram = compute_spectrogram(iq_data)

# Data normalization and augmentation
normalized_data = normalize_and_augment(spectrogram)
```

### 3. Feature Extraction
- Frequency domain feature extraction
- Time-frequency analysis
- Dynamic range adjustment (80dB)

## 📈 Training Strategy

### Recommended Training Configuration

1. **Fine-tuning Strategy** (Recommended)
```python
freeze_strategy = 'backbone'
learning_rate = 0.00007
epochs = 20-30
```

2. **Gradual Fine-tuning**
```python
freeze_strategy = 'early_features'
learning_rate = 0.0001
epochs = 25-40
```

3. **Customized Freezing**
```python
freeze_strategy = 10  # Freeze the first 10 layers
# Or
freeze_strategy = [0,1,2,3]  # Freeze the specified layers
```

## 📊 Visualization Features

The project provides rich visualization tools:

- 📈 **Training History**：Loss and accuracy curves
- 🎯 **Confusion Matrix**：Classification result visualization
- 📊 **Probability Distribution**：Confidence analysis
- 🔍 **Spectrogram**：Signal feature visualization

## 🛠️ Advanced Usage

### Custom Data Set

```python
# Prepare your data
data_dir = './your_data'
dataset = prepare_dataset(data_dir)

# Train the model
trainer = ModelTrainer(model, dataset)
trainer.train()
```

### Model Evaluation

```python
# Load the trained model
model = torch.load('./snapshot/best_model.pth')

# Evaluate on the test set
results = test_model(model, test_loader)
```

### Batch Prediction

```python
# Predict on new data
predictions = model.predict_batch(new_signals)
```


### Technical Documentation
- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](jamming_part/LICENSE) file for details.

---

<div align="center">

**⭐ If this project is helpful to you, please give us a star! ⭐**

</div>

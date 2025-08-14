import numpy as np
import torch
import random

# Import custom modules
from data_processing import prepare_dataset, dump_images_multiprocess
from model import MobileNetV2, ModelTrainer
from utils import test_model, plot_training_history, plot_confusion_matrix, plot_probability_distribution

# ============ Global Configuration Parameters ============
# Random seed setting
RANDOM_SEED = 42

# Data related parameters
DATA_DIR = 'H:/processed_data'  # Processed data directory

# Model architecture parameters
# NUM_CLASSES = 2           # Number of classes (real/spoof)
NUM_CLASSES = 6           # Number of classes (cleanDynamic, ds1, ds2, ds3, ds4, ds7)
WIDTH_MULT = 1.0          # MobileNetV2 width multiplier
DROPOUT_RATE = 0.2        # Dropout rate
FREEZE_STRATEGY = 'backbone'  # Freeze strategy: 'backbone', 'early_features', 'features', integer, or list

# Training hyperparameters
NUM_EPOCHS = 30           # Number of training epochs
LEARNING_RATE = 0.00007   # Learning rate (suitable for frozen layer training)

# File paths
MODEL_SAVE_PATH = './snapshot/best_model.pth'    # Best model save path
HISTORY_PLOT_PATH = 'training_history.png'       # Training history plot save path
HISTORY_FILE = './snapshot/training_history.txt' # Training history data file
CONFUSION_MATRIX_PATH = 'confusion_matrix.png'   # Confusion matrix plot save path
PROBABILITY_DIST_PATH = 'probability_distribution.png'  # Probability distribution plot save path
CONFIDENCE_DIST_PATH = 'probability_distribution_confidence.png'  # Confidence distribution plot save path
TEST_RESULTS_FILE = './snapshot/test_results.txt' # Test results data file

if __name__ == "__main__":
    # ============ training strategy selection guide ============
    """
    1. fine-tuning - freeze backbone (recommended):
       - freeze_strategy = 'backbone'
       - lr = 0.00007, epochs = 20-30  
       
    2. progressive fine-tuning - freeze early features:
       - freeze_strategy = 'early_features'
       - lr = 0.0001, epochs = 25-40
       
    3. custom freeze:
       - freeze_strategy = 10 (freeze first 10 layers)
       - freeze_strategy = [0,1,2,3] (freeze specified layers)
    """
    
   #  # ============ Step 1: Set random seed ============
   #  torch.manual_seed(RANDOM_SEED)
   #  np.random.seed(RANDOM_SEED)
   #  random.seed(RANDOM_SEED)
   #  print(f"Set random seed: {RANDOM_SEED}")
    
   #  # ============ Step 2: Set computing device ============
   #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   #  print(f"Using device: {device}")
    
   #  # ============ Step 3: Prepare dataset ============
   #  print("start processing raw data...")
   #  bin_files = [
   #      'H:/ds1.bin',
   #      'H:/ds2.bin', 
   #      'H:/ds3.bin',
   #      'H:/ds4.bin',
   #      'H:/ds7.bin',
   #      'H:/cleanDynamic.bin'
   #  ]
    
   #  dump_images_multiprocess(
   #      bin_files=bin_files,
   #      base_out_dir='H:/processed_data',
   #      win_sec=5,
   #      hop_sec=1,
   #      n_processes=3
   #  )

   # #  print("data processing completed!")
   #  print(f"Preparing dataset from {DATA_DIR}...")
   #  train_loader, val_loader, test_loader = prepare_dataset(DATA_DIR)
    
   #  # ============ Step 4: Initialize model ============
   #  model = MobileNetV2(
   #      num_classes=NUM_CLASSES, 
   #      width_mult=WIDTH_MULT, 
   #      dropout=DROPOUT_RATE,
   #      freeze_layers=FREEZE_STRATEGY
   #  )
   #  print(f"Model configuration:")
   #  print(f"  Number of classes: {NUM_CLASSES}")
   #  print(f"  Width multiplier: {WIDTH_MULT}")
   #  print(f"  Dropout rate: {DROPOUT_RATE}")
   #  print(f"  Freeze strategy: {FREEZE_STRATEGY}")
   #  print(f"  Learning rate: {LEARNING_RATE} (suitable for frozen layer training)")
   #  print(f"  Training epochs: {NUM_EPOCHS} (suitable for fine-tuning)")
    
   #  # ============ Step 5: Train model ============
   #  print("Starting training...")
   #  trainer = ModelTrainer(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device)
   #  model, train_losses, val_losses, val_accuracies = trainer.train()
    
   #  # ============ Step 6: Test model ============
   #  model.load_state_dict(torch.load(MODEL_SAVE_PATH))
   #  print(f"Loading best model for testing: {MODEL_SAVE_PATH}")
   #  accuracy, predictions, true_labels = test_model(model, test_loader, device)
    
   #  # ============ Step 6.5: Save test results for visualization ============
   #  trainer.save_test_results(model, test_loader, device)
    
    # ============ Step 7: Visualize results ============
    print("Generating training history plots from saved data...")
    # Read training history from txt file and plot
    plot_training_history(history_file=HISTORY_FILE, save_path=HISTORY_PLOT_PATH)
    
    print("Generating confusion matrix from saved test results...")
    # Plot confusion matrix from saved test results
    plot_confusion_matrix(test_results_file=TEST_RESULTS_FILE, save_path=CONFUSION_MATRIX_PATH)
    
    print("Generating probability distribution plots from saved test results...")
    # Plot probability distribution histograms
    plot_probability_distribution(test_results_file=TEST_RESULTS_FILE, save_path=PROBABILITY_DIST_PATH)
    
    # Get best validation accuracy from history file
    try:
        from utils import load_training_history
        history_data = load_training_history(HISTORY_FILE)
        best_val_accuracy = max(history_data['val_accuracies'])
    except:
        best_val_accuracy = 0.0
        print(f"Error: Failed to load training history from {HISTORY_FILE}")
    
    # Get test accuracy from test results file
    try:
        from utils import load_test_results
        from sklearn.metrics import accuracy_score
        test_data = load_test_results(TEST_RESULTS_FILE)
        if len(test_data) >= 2:
            true_labels, predicted_labels = test_data[:2]
            accuracy = accuracy_score(true_labels, predicted_labels)
        else:
            accuracy = 0.0
    except:
        accuracy = 0.0
        print(f"Error: Failed to load test results from {TEST_RESULTS_FILE}")
    
    # ============ Training Summary ============
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Final test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Best validation accuracy: {best_val_accuracy:.4f} ({best_val_accuracy:.2f}%)")
    print(f"Best model saved as: {MODEL_SAVE_PATH}")
    print(f"Training history plot saved as: {HISTORY_PLOT_PATH}")
    print(f"Training history data saved as: {HISTORY_FILE}")
    print(f"Confusion matrix plot saved as: {CONFUSION_MATRIX_PATH}")
    print(f"Probability distribution plot saved as: {PROBABILITY_DIST_PATH}")
    print(f"Confidence distribution plot saved as: {CONFIDENCE_DIST_PATH}")
    print(f"Test results data saved as: {TEST_RESULTS_FILE}")
    print(f"{'='*60}")



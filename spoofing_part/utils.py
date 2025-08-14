import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def test_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []
    print("\nstart testing model...")
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='testing progress')
        for data, target in test_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n{'='*50}")
    print(f"testing results")
    print(f"{'='*50}")
    print(f"testing accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\nconfusion matrix:")
    print(f"{'':>12} {'real signal':>8} {'spoof signal':>8}")
    print(f"{'real signal':>12} {cm[0,0]:>8} {cm[0,1]:>8}")
    print(f"{'spoof signal':>12} {cm[1,0]:>8} {cm[1,1]:>8}")
    print(f"\ndetailed classification report:")
    target_names = ['real signal', 'spoof signal']
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    return accuracy, all_predictions, all_labels

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_training_history(train_losses=None, val_losses=None, val_accuracies=None, history_file=None, save_path='training_history.png'):
    """
    Plot training history from either provided data or from a history file
    
    Args:
        train_losses (list, optional): Training loss values for each epoch
        val_losses (list, optional): Validation loss values for each epoch  
        val_accuracies (list, optional): Validation accuracy values for each epoch
        history_file (str, optional): Path to training history txt file. If provided, will read data from file
        save_path (str): Path to save the plot image
    
    Usage:
        # Method 1: Use data from training
        plot_training_history(train_losses, val_losses, val_accuracies)
        
        # Method 2: Read from file (recommended)
        plot_training_history(history_file='./snapshot/training_history.txt')
    """
    
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    if history_file is not None:
        # Read training history from file
        if not os.path.exists(history_file):
            raise FileNotFoundError(f"Training history file not found: {history_file}")
        
        print(f"Reading training history from: {history_file}")
        
        # Read CSV file
        df = pd.read_csv(history_file)
        
        # Extract data
        epochs = df['epoch'].tolist()
        train_losses = df['train_loss'].tolist()
        val_losses = df['val_loss'].tolist() 
        val_accuracies = df['val_accuracy'].tolist()
        
        print(f"Loaded {len(epochs)} epochs of training history")
        
    elif train_losses is not None and val_losses is not None and val_accuracies is not None:
        # Use provided data
        epochs = list(range(1, len(train_losses) + 1))
        print(f"Using provided training data with {len(epochs)} epochs")
        
    else:
        # Try to read from default location
        default_history_file = './snapshot/training_history.txt'
        if os.path.exists(default_history_file):
            print(f"No data provided, reading from default location: {default_history_file}")
            return plot_training_history(history_file=default_history_file, save_path=save_path)
        else:
            raise ValueError("Either provide training data (train_losses, val_losses, val_accuracies) or specify history_file path")
    
    # Apply smoothing
    smooth_train_losses = smooth_curve(train_losses)
    smooth_val_losses = smooth_curve(val_losses)
    smooth_val_accuracies = smooth_curve(val_accuracies)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Define custom blue color palette
    deep_blue = '#071E55'      # Deepest blue for training loss
    medium_blue = '#2046A1'    # Medium blue for validation loss  
    light_blue = '#3963C8'     # Light blue for validation accuracy
    accent_blue = '#3868D9'    # Accent blue for highlights
    
    # Loss curve
    ax1.plot(epochs, smooth_train_losses, label='Training Loss', color=deep_blue, linewidth=3)
    ax1.plot(epochs, smooth_val_losses, label='Validation Loss', color=medium_blue, linewidth=3)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, smooth_val_accuracies, label='Validation Accuracy', color=light_blue, linewidth=3)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add best accuracy line
    if val_accuracies:
        best_acc = max(val_accuracies)
        best_epoch = val_accuracies.index(best_acc) + 1
        ax2.axhline(y=best_acc, color=accent_blue, linestyle='--', alpha=0.8, linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved as: {save_path}")
    
    # Display the plot
    plt.show()
    
    return fig

def load_training_history(history_file='./snapshot/training_history.txt'):
    """
    Load training history from file
    
    Args:
        history_file (str): Path to training history txt file
        
    Returns:
        dict: Dictionary containing epochs, train_losses, val_losses, val_accuracies
    """
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"Training history file not found: {history_file}")
    
    df = pd.read_csv(history_file)
    
    return {
        'epochs': df['epoch'].tolist(),
        'train_losses': df['train_loss'].tolist(),
        'val_losses': df['val_loss'].tolist(),
        'val_accuracies': df['val_accuracy'].tolist()
    } 

def load_test_results(test_results_file='./snapshot/test_results.txt'):
    """
    Load test results from file
    
    Args:
        test_results_file (str): Path to test results txt file
        
    Returns:
        tuple: (true_labels, predicted_labels) or (true_labels, predicted_labels, probabilities)
    """
    if not os.path.exists(test_results_file):
        raise FileNotFoundError(f"Test results file not found: {test_results_file}")
    
    df = pd.read_csv(test_results_file)
    
    # Check if file contains probability columns
    if 'prob_real' in df.columns and 'prob_spoof' in df.columns:
        # New format with probabilities
        probabilities = df[['prob_real', 'prob_spoof']].values
        return df['true_label'].tolist(), df['predicted_label'].tolist(), probabilities
    else:
        # Old format without probabilities (backward compatibility)
        return df['true_label'].tolist(), df['predicted_label'].tolist()

def plot_probability_distribution(test_results_file=None, save_path='probability_distribution.png', class_names=None):
    """
    Plot prediction probability distribution histogram
    
    Args:
        test_results_file (str, optional): Path to test results file. If None, uses default path
        save_path (str): Path to save the probability distribution plot
        class_names (list, optional): List of class names. If None, uses default ['Real Signal', 'Spoof Signal']
    
    Returns:
        matplotlib.figure.Figure: The probability distribution figure
    """
    
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Set default file path if not provided
    if test_results_file is None:
        test_results_file = './snapshot/test_results.txt'
    
    # Set default class names if not provided
    if class_names is None:
        class_names = ['Real Signal', 'Spoof Signal']
    
    try:
        # Load test results from file
        print(f"Loading test results from: {test_results_file}")
        
        # Try to load with probabilities
        test_data = load_test_results(test_results_file)
        if len(test_data) == 3:
            true_labels, predicted_labels, probabilities = test_data
            print(f"Loaded {len(true_labels)} test samples with probability data")
        else:
            print("Error: Probability data not found in test results file.")
            print("Please retrain the model to generate probability data.")
            return None
        
        # Convert to numpy arrays for easier processing
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        probabilities = np.array(probabilities)
        
        # Define blue color palette matching other plots
        deep_blue = '#071E55'
        medium_blue = '#2046A1'
        light_blue = '#3963C8'
        accent_blue = '#3868D9'
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Probability distribution for Real Signal class (class 0)
        real_samples_mask = (true_labels == 0)
        real_probs = probabilities[real_samples_mask, 0]  # Probability of being real
        
        ax1.hist(real_probs, bins=30, alpha=0.7, color=deep_blue, edgecolor='white', linewidth=1)
        ax1.set_title(f'Real {class_names[0]} Samples\nPredicted Probability Distribution', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted Probability (Real Signal)', fontsize=10)
        ax1.set_ylabel('Number of Samples', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Decision Threshold')
        ax1.legend()
        

        
        # 2. Probability distribution for Spoof Signal class (class 1)
        spoof_samples_mask = (true_labels == 1)
        spoof_probs = probabilities[spoof_samples_mask, 1]  # Probability of being spoof
        
        ax2.hist(spoof_probs, bins=30, alpha=0.7, color=medium_blue, edgecolor='white', linewidth=1)
        ax2.set_title(f'Real {class_names[1]} Samples\nPredicted Probability Distribution', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted Probability (Spoof Signal)', fontsize=10)
        ax2.set_ylabel('Number of Samples', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Decision Threshold')
        ax2.legend()
        

        
        # 3. Combined probability distribution (overlapped)
        ax3.hist(real_probs, bins=30, alpha=0.6, color=deep_blue, label=f'True {class_names[0]}', 
                edgecolor='white', linewidth=1)
        ax3.hist(1 - spoof_probs, bins=30, alpha=0.6, color=light_blue, label=f'True {class_names[1]}', 
                edgecolor='white', linewidth=1)
        ax3.set_title('Overlapped Probability Distribution\n(Both classes on Real Signal probability)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Predicted Probability (Real Signal)', fontsize=10)
        ax3.set_ylabel('Number of Samples', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Decision Threshold')
        ax3.legend()
        
        # 4. Confidence analysis
        all_probs = np.max(probabilities, axis=1)  # Maximum probability (confidence)
        correct_predictions = (predicted_labels == true_labels)
        
        # Separate confidence for correct and incorrect predictions
        correct_confidence = all_probs[correct_predictions]
        incorrect_confidence = all_probs[~correct_predictions]
        
        ax4.hist(correct_confidence, bins=20, alpha=0.7, color=accent_blue, 
                label=f'Correct Predictions ({len(correct_confidence)})', edgecolor='white', linewidth=1)
        ax4.hist(incorrect_confidence, bins=20, alpha=0.7, color='red', 
                label=f'Incorrect Predictions ({len(incorrect_confidence)})', edgecolor='white', linewidth=1)
        ax4.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Maximum Probability (Confidence)', fontsize=10)
        ax4.set_ylabel('Number of Samples', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        

        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability distribution plot saved as: {save_path}")
        
        # Display the plot
        plt.show()
        
        return fig
        
    except FileNotFoundError:
        print(f"Error: Test results file not found at: {test_results_file}")
        print("Please run the model evaluation first to generate test results.")
        return None
        
    except Exception as e:
        print(f"Error plotting probability distribution: {e}")
        return None

def plot_confusion_matrix(test_results_file=None, save_path='confusion_matrix.png', class_names=None):
    """
    Plot confusion matrix from saved test results
    
    Args:
        test_results_file (str, optional): Path to test results file. If None, uses default path
        save_path (str): Path to save the confusion matrix plot
        class_names (list, optional): List of class names. If None, uses default ['Real Signal', 'Spoof Signal']
    
    Returns:
        matplotlib.figure.Figure: The confusion matrix figure
    """
    
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Set default file path if not provided
    if test_results_file is None:
        test_results_file = './snapshot/test_results.txt'
    
    # Set default class names if not provided
    if class_names is None:
        class_names = ['Real Signal', 'Spoof Signal']
    
    try:
        # Load test results from file
        print(f"Loading test results from: {test_results_file}")
        
        # Load data (handle both old and new formats)
        test_data = load_test_results(test_results_file)
        if len(test_data) == 3:
            true_labels, predicted_labels, probabilities = test_data
        else:
            true_labels, predicted_labels = test_data
        
        print(f"Loaded {len(true_labels)} test samples")
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Create figure - use only matplotlib to avoid seaborn dependency
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Create a custom blue colormap
        from matplotlib.colors import LinearSegmentedColormap
        blues = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
        blue_cmap = LinearSegmentedColormap.from_list('custom_blues', blues, N=256)
        
        # Plot confusion matrix heatmap manually
        im = ax.imshow(cm, interpolation='nearest', cmap=blue_cmap)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Samples', fontsize=12)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}',
                       ha="center", va="center", fontsize=14, fontweight='bold',
                       color="white" if cm[i, j] > thresh else "black")
        
        # Customize the plot
        ax.set_title(f'Confusion Matrix\nOverall Accuracy: {accuracy:.1%}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        
        # Set tick labels
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)
        
        # Add grid lines
        ax.set_xlim(-0.5, len(class_names)-0.5)
        ax.set_ylim(-0.5, len(class_names)-0.5)
        

        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved as: {save_path}")
        
        # Display the plot
        plt.show()
        return fig
        
    except FileNotFoundError:
        print(f"Error: Test results file not found at: {test_results_file}")
        print("Please run the model evaluation first to generate test results.")
        return None
        
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        return None 



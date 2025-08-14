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
    print(cm)
    print(f"\ndetailed classification report:")
    target_names = [
        'Real Signal',
        'RF Switch', 
        'Over-powered Time-Push',
        'Matched-power Time-Push',
        'Matched-power Position-Push',
        'Seamless Matched-power Time-Push'
    ]
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


def plot_confidence_distribution(probabilities, predicted_labels, true_labels, thresholds, save_path):
    """
    Plot overall prediction confidence distribution
    
    Args:
        probabilities (np.array): Array of prediction probabilities
        predicted_labels (list): List of predicted labels
        true_labels (list): List of true labels
        thresholds (list): List of threshold values
        save_path (str): Path to save the confidence plot
        
    Returns:
        matplotlib.figure.Figure: The confidence distribution figure
    """
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Create single figure for confidence analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate confidence (maximum probability)
    all_probs = np.max(probabilities, axis=1)
    correct_predictions = (np.array(predicted_labels) == np.array(true_labels))
    
    # Separate confidence for correct and incorrect predictions
    correct_confidence = all_probs[correct_predictions]
    incorrect_confidence = all_probs[~correct_predictions]
    
    # Plot histograms with different colors for correct/incorrect
    ax.hist(correct_confidence, bins=25, alpha=0.7, color='#2E8B57', 
            label=f'Correct Predictions ({len(correct_confidence)})', 
            edgecolor='white', linewidth=1)
    ax.hist(incorrect_confidence, bins=25, alpha=0.7, color='#DC143C', 
            label=f'Incorrect Predictions ({len(incorrect_confidence)})', 
            edgecolor='white', linewidth=1)
    
    # Add confidence threshold lines
    threshold_colors = ['red', 'orange', 'green']
    threshold_styles = ['--', '-.', ':']
    
    for thresh, color, style in zip(thresholds, threshold_colors, threshold_styles):
        ax.axvline(x=thresh, color=color, linestyle=style, linewidth=2, alpha=0.8,
                  label=f'Threshold {thresh}')
    
    # Set figure title instead of axis title
    fig.suptitle('Overall Prediction Confidence Distribution', fontsize=16, fontweight='bold', y=0.96)
    
    ax.set_xlabel('Maximum Probability (Confidence)', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Position both legends between title and chart
    from matplotlib.patches import Patch
    
    # Create prediction legend elements
    pred_patches = [Patch(color='#2E8B57', alpha=0.7), Patch(color='#DC143C', alpha=0.7)]
    pred_labels = [f'Correct Predictions ({len(correct_confidence)})',
                   f'Incorrect Predictions ({len(incorrect_confidence)})']
    
    # Create threshold legend elements
    threshold_lines = []
    threshold_labels = []
    for thresh, color, style in zip(thresholds, threshold_colors, threshold_styles):
        line = plt.Line2D([0], [0], color=color, linestyle=style, linewidth=2)
        threshold_lines.append(line)
        threshold_labels.append(f'Threshold {thresh}')
    
    # Add prediction legend below title (first row)
    fig.legend(pred_patches, pred_labels, 
               loc='upper center', bbox_to_anchor=(0.5, 0.87),
               ncol=2, fontsize=10, frameon=True)
    
    # Add threshold legend below prediction legend (second row)  
    fig.legend(threshold_lines, threshold_labels, 
               loc='upper center', bbox_to_anchor=(0.5, 0.92),
               ncol=3, fontsize=10, frameon=True)
    
    # Add comprehensive threshold statistics
    threshold_stats = []
    total_samples = len(all_probs)
    overall_accuracy = np.mean(correct_predictions)
    
    threshold_stats.append(f"Overall Accuracy: {overall_accuracy:.3f}")
    threshold_stats.append("─" * 25)
    
    for thresh in thresholds:
        high_conf_mask = all_probs >= thresh
        high_conf_count = np.sum(high_conf_mask)
        if high_conf_count > 0:
            high_conf_accuracy = np.mean(correct_predictions[high_conf_mask])
            coverage = high_conf_count / total_samples
            threshold_stats.append(f"Threshold {thresh}:")
            threshold_stats.append(f"  Accuracy: {high_conf_accuracy:.3f}")
            threshold_stats.append(f"  Coverage: {coverage:.3f}")
        else:
            threshold_stats.append(f"Threshold {thresh}: No samples")
    
    # Add text box with threshold statistics (top left)
    if threshold_stats:
        stats_text = '\n'.join(threshold_stats)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Adjust layout with space for title and legends above chart
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confidence distribution plot saved as: {save_path}")
    
    # Display the plot
    plt.show()
    
    return fig


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
    prob_columns = [col for col in df.columns if col.startswith('prob_')]
    
    if prob_columns:
        # Format with probabilities
        probabilities = df[prob_columns].values
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
        class_names (list, optional): List of class names. If None, uses default 6-class names
    
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
        class_names = [
            'Real Signal',
            'RF Switch',
            'Over-powered Time-Push',
            'Matched-power Time-Push',
            'Matched-power Position-Push',
            'Seamless Matched-power Time-Push'
        ]
    
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
        
        # Check if probability data matches expected number of classes
        expected_classes = len(class_names)
        actual_classes = probabilities.shape[1]
        
        if actual_classes != expected_classes:
            print(f"Warning: Probability data mismatch!")
            print(f"Expected {expected_classes} classes, but got {actual_classes} classes in probability data.")
            print(f"This usually means the test results were generated with a different model configuration.")
            print(f"Please retrain the model with {expected_classes} classes to generate proper probability data.")
            return None
        
        # Set fixed thresholds
        thresholds = [0.5, 0.7, 0.9]
        
        # Define color palette based on reference image
        color1 = '#015C92'  # Deep blue
        color2 = '#004D7A'  # Dark blue (darker than color1)
        color3 = '#2D82B5'  # Medium-deep blue  
        color4 = '#53A7D8'  # Medium blue
        color5 = '#88CDF6'  # Light blue
        color6 = '#BCE6FF'  # Very light blue
        
        
        # Get number of classes
        num_classes = len(class_names)
        
        # Create figure: 2x3 layout for 6 classes only
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Define colors for each class
        colors = [color1, color2, color3, color4, color5, color6]
        
        # Initialize legend handles for global legend
        legend_handles = None
        
        # Plot "correct class probability" for each true class
        class_stats = {}
        for class_idx in range(num_classes):
            ax = axes[class_idx]
            
            # Get samples that truly belong to this class
            true_class_mask = (true_labels == class_idx)
            class_count = np.sum(true_class_mask)
            
            if class_count > 0:
                # Get predicted probabilities for the CORRECT class (class_idx) 
                # for samples that truly belong to this class
                correct_class_probs = probabilities[true_class_mask, class_idx]
                class_stats[class_idx] = np.mean(correct_class_probs)
                
                # Plot histogram of correct class probabilities
                ax.hist(correct_class_probs, bins=20, alpha=0.7, color=colors[class_idx % len(colors)], 
                       edgecolor='white', linewidth=1)
                
                # Add threshold lines
                threshold_colors = ['red', 'orange', 'green']
                threshold_styles = ['--', '-.', ':']
                
                for thresh, color, style in zip(thresholds, threshold_colors, threshold_styles):
                    ax.axvline(x=thresh, color=color, linestyle=style, linewidth=2, alpha=0.8,
                              label=f'Threshold {thresh}')
                
                # Calculate accuracy at different thresholds
                thresh_stats = []
                for thresh in thresholds:
                    high_conf_mask = correct_class_probs >= thresh
                    if np.sum(high_conf_mask) > 0:
                        coverage = np.mean(high_conf_mask)
                        thresh_stats.append(f"{thresh}: {coverage:.2f}")
                
                # Add statistics text
                if thresh_stats:
                    stats_text = "Coverage:\n" + "\n".join(thresh_stats)
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'{class_names[class_idx]}\n(n={class_count})', 
                           fontsize=11, fontweight='bold')
                ax.set_xlabel('Correct Class Probability', fontsize=9)
                ax.set_ylabel('Number of Samples', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                
                # Store legend handles for global legend (only for first subplot with data)
                if class_idx == 0 and legend_handles is None:
                    legend_handles = ax.get_lines()[-3:]  # Get the last 3 lines (threshold lines)
            else:
                # No samples for this class
                ax.text(0.5, 0.5, f'No {class_names[class_idx]}\nsamples in test set', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                ax.set_title(f'{class_names[class_idx]}\n(n=0)', 
                           fontsize=11, fontweight='bold')
                ax.set_xlim(0, 1)
                ax.set_xlabel('Correct Class Probability', fontsize=9)
                ax.set_ylabel('Number of Samples', fontsize=9)
        

        # Add main title for the entire figure
        plt.suptitle('Probability Distribution of Correct Class Predictions', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Add global legend for thresholds below the title
        if legend_handles is not None:
            legend_labels = [f'Threshold {thresh}' for thresh in thresholds]
            fig.legend(legend_handles, legend_labels, 
                      loc='upper center', bbox_to_anchor=(0.5, 0.93), 
                      ncol=3, fontsize=10, frameon=True, 
                      fancybox=True, shadow=True)
        
        # Adjust layout with padding to accommodate title and legend
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability distribution plot saved as: {save_path}")
        
        # Display the plot
        plt.show()
        
        # Create separate confidence plot
        confidence_save_path = save_path.replace('.png', '_confidence.png')
        fig_conf = plot_confidence_distribution(probabilities, predicted_labels, true_labels, 
                                               thresholds, confidence_save_path)
        
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
        class_names (list, optional): List of class names. If None, uses default 6-class names
    
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
        # class_names = [
        #     'Real Signal',
        #     'RF Switch',
        #     'Over-powered Time-Push',
        #     'Matched-power Time-Push',
        #     'Matched-power Position-Push',
        #     'Seamless Matched-power Time-Push'
        # ]
        class_names = [
            'Real',
            'RFS',
            'OPTP',
            'MPTP',
            'MPPP',
            'SMTP'
        ]
    
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



# Utility functions for AMD Detection Model

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import os
import json


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                    ha='center', va='center', fontsize=9, color='red')
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate precision, recall, and F1-score for each class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary containing metrics
    """
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision': {},
        'recall': {},
        'f1': {},
        'support': {}
    }
    
    for i, class_name in enumerate(class_names):
        metrics['precision'][class_name] = float(precision[i])
        metrics['recall'][class_name] = float(recall[i])
        metrics['f1'][class_name] = float(f1[i])
        metrics['support'][class_name] = int(support[i])
    
    # Macro averages
    metrics['macro_precision'] = float(np.mean(precision))
    metrics['macro_recall'] = float(np.mean(recall))
    metrics['macro_f1'] = float(np.mean(f1))
    
    # Weighted averages
    metrics['weighted_precision'] = float(np.average(precision, weights=support))
    metrics['weighted_recall'] = float(np.average(recall, weights=support))
    metrics['weighted_f1'] = float(np.average(f1, weights=support))
    
    return metrics


def save_checkpoint(model, optimizer, epoch, val_acc, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        val_acc: Validation accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
    
    Returns:
        Tuple of (epoch, val_acc)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resumed from epoch {epoch} with validation accuracy {val_acc:.4f}")
    
    return epoch, val_acc


def predict_image(model, image_path, device='cuda', class_names=None):
    """
    Predict class for a single image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to image file
        device: Device to run inference on
        class_names: List of class names
    
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    from PIL import Image
    from torchvision import transforms
    import config
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    all_probs = probabilities.cpu().numpy()[0]
    
    if class_names:
        predicted_label = class_names[predicted_class]
    else:
        predicted_label = predicted_class
    
    return predicted_label, confidence_score, all_probs


def visualize_predictions(model, data_loader, device, class_names, num_images=16, save_path=None):
    """
    Visualize model predictions on a batch of images
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader
        device: Device to run inference on
        class_names: List of class names
        num_images: Number of images to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    
    images_shown = 0
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    break
                
                # Denormalize image
                img = inputs[i].cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Plot
                axes[images_shown].imshow(img)
                true_label = class_names[labels[i].item()]
                pred_label = class_names[preds[i].item()]
                color = 'green' if labels[i] == preds[i] else 'red'
                axes[images_shown].set_title(f'True: {true_label}\nPred: {pred_label}', 
                                            color=color, fontsize=10, fontweight='bold')
                axes[images_shown].axis('off')
                
                images_shown += 1
            
            if images_shown >= num_images:
                break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    
    plt.close()

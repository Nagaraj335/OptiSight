# Training script for AMD Detection Model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import copy
from datetime import datetime
import json
import os

import config
from data_loader import get_data_loaders
from model import get_model
from utils import (
    plot_training_history, 
    plot_confusion_matrix, 
    calculate_metrics,
    save_checkpoint,
    load_checkpoint
)


class AMDTrainer:
    """Trainer class for AMD detection model"""
    
    def __init__(self, model_name='resnet50', device=None):
        """
        Initialize the trainer
        
        Args:
            model_name: Name of the model architecture
            device: Device to train on (cuda/cpu)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = get_model(model_name, device=self.device)
        self.model_name = model_name
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=config.REDUCE_LR_PATIENCE
        )
        
        # Get data loaders
        self.train_loader, self.val_loader = get_data_loaders()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_wts = None
        self.epochs_no_improve = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.val_loader.dataset)
        
        return epoch_loss, epoch_acc.item(), np.array(all_preds), np.array(all_labels)
    
    def train(self, num_epochs=None):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train (defaults to config.EPOCHS)
        """
        if num_epochs is None:
            num_epochs = config.EPOCHS
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Print epoch results
            print(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                print(f"  ✓ Validation accuracy improved from {self.best_val_acc:.4f} to {val_acc:.4f}")
                self.best_val_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(config.MODEL_DIR, f'best_model_{self.model_name}.pth')
                save_checkpoint(self.model, self.optimizer, epoch, val_acc, checkpoint_path)
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 80)
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)
        
        # Final evaluation
        self.evaluate()
        
        # Plot results
        self.plot_results()
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def evaluate(self):
        """Evaluate the model on validation set"""
        print("\nFinal Evaluation:")
        print("=" * 80)
        
        val_loss, val_acc, val_preds, val_labels = self.validate_epoch()
        
        # Calculate detailed metrics
        metrics = calculate_metrics(val_labels, val_preds, config.CLASS_NAMES)
        
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"\nPer-class metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 51)
        
        for class_name in config.CLASS_NAMES:
            print(f"{class_name:<15} "
                  f"{metrics['precision'][class_name]:<12.4f} "
                  f"{metrics['recall'][class_name]:<12.4f} "
                  f"{metrics['f1'][class_name]:<12.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(val_labels, val_preds, config.CLASS_NAMES, 
                            save_path=os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'))
        
        # Save metrics
        metrics_path = os.path.join(config.RESULTS_DIR, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nMetrics saved to {metrics_path}")
        
        return metrics
    
    def plot_results(self):
        """Plot training history"""
        plot_training_history(
            self.history,
            save_path=os.path.join(config.RESULTS_DIR, 'training_history.png')
        )
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(config.RESULTS_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {history_path}")


def main():
    """Main training function"""
    print("=" * 80)
    print("AMD Detection Model Training")
    print("=" * 80)
    
    # Create trainer
    trainer = AMDTrainer(model_name='resnet50')
    
    # Train the model
    history = trainer.train()
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

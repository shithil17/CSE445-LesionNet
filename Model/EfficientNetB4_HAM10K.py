import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class EfficientNetB4GradualUnfreezingTrainer:
    def __init__(self, data_dir, batch_size=64, num_epochs=100, learning_rate=0.001):
        """
        EfficientNetB4 with gradual unfreezing:
        - Epochs 1-4: Train only head (frozen backbone)
        - Epoch 5: Unfreeze blocks 6, 7, 8 (deepest blocks)
        - Epoch 25: Unfreeze blocks 4, 5 (deeper blocks)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_classes = 7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['NV', 'MEL', 'BKL', 'BCC', 'AKIEC', 'VASC', 'DF']
        
        print(f"Using device: {self.device}")
        print("="*60)
        print("EfficientNetB4 with Gradual Unfreezing (Blocks@10, Blocks@25)")
        print("="*60)
        
    def create_dataloaders(self):
        """Create train/val dataloaders with 80-20 split and manual class order"""
        # EfficientNetB4 optimal input size is 380x380, but using 224x224 for consistency
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=train_transform)
        
        # Update class names and count to match ImageFolder's found classes
        self.class_names = full_dataset.classes
        self.num_classes = len(self.class_names)
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        val_dataset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Classes (ImageFolder order): {self.class_names}")
        
        return train_loader, val_loader
    
    def build_model(self):
        """Build EfficientNetB4 with gradual unfreezing capability"""
        print("\nBuilding EfficientNetB4 model...")
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        
        # Initially freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace classifier head
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, self.num_classes)
        )
        
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,} (paper: ~19M)")
        print(f"Trainable parameters: {trainable_params:,} (initially)")
        
        return model
    
    def unfreeze_blocks(self, model, block_indices, optimizer, lr_factor=0.1):
        """
        Unfreeze specific EfficientNet blocks and add to optimizer
        block_indices: list of block numbers to unfreeze (e.g., [29, 30, 31])
        """
        parameters_to_add = []
        unfrozen_count = 0
        
        for name, param in model.named_parameters():
            # Check if parameter belongs to any of the specified blocks
            for block_idx in block_indices:
                block_name = f'features.{block_idx}.'  # EfficientNet blocks are in features.{index}
                if block_name in name:
                    param.requires_grad = True
                    parameters_to_add.append(param)
                    unfrozen_count += param.numel()
                    break
        
        if parameters_to_add:
            optimizer.add_param_group({
                'params': parameters_to_add,
                'lr': self.learning_rate * lr_factor
            })
            print(f"   → Unfrozen {unfrozen_count:,} parameters in blocks {block_indices}")
            return unfrozen_count
        return 0
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", leave=False, ncols=100)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, model, val_loader, criterion, epoch):
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]  ", leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets)
    
    def plot_training_curves(self, history):
        """Plot and save training curves"""
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training Accuracy
        ax1.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax1.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax1.set_title('Training & Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Training Loss
        ax2.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax2.set_title('Training & Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Learning Rate
        if 'lr' in history:
            ax3.plot(epochs, history['lr'], 'g-')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True)
        
        # Validation Accuracy (zoomed)
        ax4.plot(epochs, history['val_acc'], 'r-')
        ax4.set_title('Validation Accuracy (Detailed)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Training curves saved as 'training_curves.png'")
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - EfficientNetB4 (Epoch {epoch+1})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'efficientnetb4_confusion_matrix_epoch_{epoch+1}.png', dpi=300)
        plt.close()
    
    def train(self):
        """Full training pipeline with gradual unfreezing"""
        train_loader, val_loader = self.create_dataloaders()
        model = self.build_model()
        criterion = nn.CrossEntropyLoss()
        
        # Initial optimizer (only head)
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7, verbose=True
        )
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 10
        
        print("\n" + "="*60)
        print("Training with Gradual Unfreezing")
        print("="*60)
        print(f"Epochs 1-4:   Train head only")
        print(f"Epochs 5-24: Train head + blocks 6-8 (unfrozen at epoch 5)")
        print(f"Epochs 25+:   Train head + blocks 4-8 (unfrozen at epoch 25)")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # ===== GRADUAL UNFREEZING LOGIC =====
            if epoch == 5:
                print(f"\n🔓 UNFREEZING blocks 6-8 at epoch {epoch+1}...")
                # EfficientNetB4 has blocks 0-8 (features.0 to features.8)
                # Unfreeze blocks 6, 7, 8 (deepest blocks)
                self.unfreeze_blocks(model, [6, 7, 8], optimizer, lr_factor=0.1)
                
            if epoch == 25:
                print(f"\n🔓 UNFREEZING blocks 4-5 at epoch {epoch+1}...") 
                # Unfreeze blocks 4, 5
                self.unfreeze_blocks(model, [4, 5], optimizer, lr_factor=0.1)
            
            # Training
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            torch.save(model.state_dict(), f'efficientnetb4_checkpoint_model_epoch{epoch}.pth')

            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(model, val_loader, criterion, epoch)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # ===== ENHANCED EPOCH SUMMARY =====
            print("\n" + "="*80)
            print(f"📊 EPOCH {epoch+1}/{self.num_epochs} SUMMARY")
            print("="*80)
            print(f"🎯 Training   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
            print(f"✅ Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
            print(f"📈 Best Val Acc: {best_val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # ===== MODEL SAVING / NO IMPROVEMENT LOGIC =====
            if val_acc > best_val_acc:
                improvement = val_acc - best_val_acc
                best_val_acc = val_acc
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, 'efficientnetb4_best_model.pth')
                
                print(f"💾 SAVING MODEL → Epoch {epoch+1} | New Best: {val_acc:.2f}% (+{improvement:.2f}%)")
                self.plot_confusion_matrix(val_targets, val_preds, epoch)
                
            else:
                patience_counter += 1
                print(f"⏳ NO IMPROVEMENT → Epoch {epoch+1} | Patience: {patience_counter}/{early_stop_patience}")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\n🛑 EARLY STOPPING triggered at epoch {epoch+1}")
                break
        
        total_training_time = time.time() - start_time
        
        # ===== FINAL RESULTS =====
        print("\n" + "="*60)
        print("🏁 TRAINING COMPLETED")
        print("="*60)
        print(f"Total time: {total_training_time:.2f} seconds")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        
        # Load best model
        checkpoint = torch.load('efficientnetb4_best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['epoch']
        
        # Final evaluation
        final_loss, final_acc, final_preds, final_targets = self.validate(model, val_loader, criterion, best_epoch)
        
        # Plot training curves
        self.plot_training_curves(history)
        
        # Print final metrics
        print("\n" + "="*60)
        print("📋 FINAL CLASSIFICATION REPORT")
        print("="*60)
        report = classification_report(
            final_targets, final_preds,
            target_names=self.class_names,
            digits=4,
            output_dict=True
        )
        
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-"*80)
        for class_name in self.class_names:
            metrics = report[class_name]
            print(f"{class_name:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {metrics['support']:<10}")
        
        print("-"*80)
        print(f"{'Macro Avg':<10} {report['macro avg']['precision']:<12.4f} "
              f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
        print(f"{'Weighted Avg':<10} {report['weighted avg']['precision']:<12.4f} "
              f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}")
        
        print("\n" + "="*60)
        print(f"🎯 OVERALL ACCURACY: {final_acc:.2f}%")
        print("="*60)
        
        # Save metrics
        metrics = {
            'final_accuracy': final_acc,
            'best_accuracy': best_val_acc,
            'total_training_time': total_training_time,
            'epochs_trained': len(history['train_loss']),
            'best_epoch': best_epoch,
            'history': history
        }
        torch.save(metrics, 'efficientnetb4_training_metrics.pth')
        
        return model, history, metrics

if __name__ == "__main__":
    trainer = EfficientNetB4GradualUnfreezingTrainer(
        data_dir='../Dattaset',
        batch_size=64,
        num_epochs=50,
        learning_rate=0.001
    )
    model, history, metrics = trainer.train()
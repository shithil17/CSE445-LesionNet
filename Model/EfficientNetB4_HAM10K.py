import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class EfficientNetB4GradualUnfreezingTrainer:
    def __init__(self, data_dir, val_data_dir='HAM10000_split/val', test_data_dir='HAM10000_split/test', batch_size=64, num_epochs=100, learning_rate=0.001, sampler_mode='sqrt'):
        """
        EfficientNetB4 with gradual unfreezing:
        - Epochs 1-4: Train only head (frozen backbone)
        - Epoch 5: Unfreeze blocks 6, 7, 8 (deepest blocks)
        - Epoch 25: Unfreeze blocks 4, 5 (deeper blocks)
        - data_dir = raw TRAIN split (HAM10000_split/train); augmentation is on-the-fly
        - val_data_dir = raw VAL split, used only for best-model / early-stop / LR step
        - test_data_dir = raw TEST split, evaluated exactly once at the end
        - sampler_mode = 'sqrt' (1/sqrt(count), default) | 'inverse' (1/count) | 'none' (plain shuffle)
        """
        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.sampler_mode = sampler_mode
        self.num_classes = 7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['NV', 'MEL', 'BKL', 'BCC', 'AKIEC', 'VASC', 'DF']
        
        print(f"Using device: {self.device}")
        print("="*60)
        print("EfficientNetB4 with Gradual Unfreezing (Blocks@10, Blocks@25)")
        print("="*60)
        
    def create_dataloaders(self):
        """Train on the raw train split with on-the-fly augmentation; early-stop on raw val; final eval on raw test."""
        # EfficientNetB4 optimal input size is 380x380, but using 224x224 for consistency
        # Train: on-the-fly augmentation, conservative geometric only (no vertical flip, no black fill)
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(
                224,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.ImageFolder(root=self.data_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=self.val_data_dir, transform=val_transform)
        test_dataset = datasets.ImageFolder(root=self.test_data_dir, transform=val_transform)
        
        # Update class names and count to match ImageFolder's found classes
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)
        
        for name, ds in [("val", val_dataset), ("test", test_dataset)]:
            if self.class_names != ds.classes:
                raise ValueError(
                    f"Train/{name} class folders differ: {self.class_names} vs {ds.classes}"
                )
        
        # Moderated class-balanced sampling (WeightedRandomSampler replaces shuffle)
        class_counts = np.bincount(train_dataset.targets, minlength=self.num_classes)
        if self.sampler_mode == 'sqrt':
            class_weights = 1.0 / np.sqrt(class_counts)
        elif self.sampler_mode == 'inverse':
            class_weights = 1.0 / class_counts
        elif self.sampler_mode == 'none':
            class_weights = None
        else:
            raise ValueError(f"Unknown sampler_mode: {self.sampler_mode}")
        
        if class_weights is not None:
            sample_weights = np.array([class_weights[t] for t in train_dataset.targets])
            train_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(train_dataset), replacement=True
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Classes (ImageFolder order): {self.class_names}")
        print(f"Train class counts: {dict(zip(self.class_names, class_counts))}")
        if class_weights is not None:
            print(f"Sampler mode: {self.sampler_mode} (relative weights: "
                  f"{dict(zip(self.class_names, (class_weights / class_weights.min()).round(2)))})")
        
        return train_loader, val_loader, test_loader
    
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
        all_probs = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]  ", leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                probabilities = torch.softmax(output, dim=1)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return (
            epoch_loss,
            epoch_acc,
            np.array(all_preds),
            np.array(all_targets),
            np.array(all_probs)
        )
    
    def plot_training_curves(self, history):
        """Plot and save training curves"""
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training Accuracy
        ax1.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax1.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax1.axvline(x=6, linestyle='--', color='gray', label='Unfreeze Blocks 6-8')
        ax1.axvline(x=26, linestyle='--', color='black', label='Unfreeze Blocks 4-5')
        ax1.set_title('Training & Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Training Loss
        ax2.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax2.axvline(x=6, linestyle='--', color='gray', label='Unfreeze Blocks 6-8')
        ax2.axvline(x=26, linestyle='--', color='black', label='Unfreeze Blocks 4-5')
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
        ax4.axvline(x=6, linestyle='--', color='gray', label='Unfreeze Blocks 6-8')
        ax4.axvline(x=26, linestyle='--', color='black', label='Unfreeze Blocks 4-5')
        ax4.set_title('Validation Accuracy (Detailed)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Training curves saved as 'training_curves.png'")
    
    def _plot_confusion_matrix(self, y_true, y_pred, title, filename):
        """Shared confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.num_classes))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """Plot confusion matrix for an epoch"""
        self._plot_confusion_matrix(
            y_true, y_pred,
            f'Confusion Matrix - EfficientNetB4 (Epoch {epoch+1})',
            f'efficientnetb4_confusion_matrix_epoch_{epoch+1}.png'
        )
    
    def plot_final_confusion_matrix(self, y_true, y_pred):
        """Plot the final confusion matrix from the best model"""
        self._plot_confusion_matrix(
            y_true, y_pred,
            'Confusion Matrix - EfficientNetB4 (Final)',
            'confusion_matrix.png'
        )
        print("📊 Confusion matrix saved as 'confusion_matrix.png'")
    
    def plot_roc_curves(self, y_true, probabilities):
        """Plot one-vs-rest ROC curves for all classes"""
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            binary_targets = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_targets, probabilities[:, i])
            roc_auc = roc_auc_score(binary_targets, probabilities[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - EfficientNetB4')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 ROC curves saved as 'roc_curves.png'")
    
    def plot_pr_curves(self, y_true, probabilities):
        """Plot one-vs-rest Precision-Recall curves and return per-class PR-AUC"""
        pr_auc_per_class = {}
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            binary_targets = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(binary_targets, probabilities[:, i])
            pr_auc = average_precision_score(binary_targets, probabilities[:, i])
            pr_auc_per_class[class_name] = pr_auc
            plt.plot(recall, precision, label=f'{class_name} (AP={pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - EfficientNetB4')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Precision-Recall curves saved as 'precision_recall_curves.png'")
        return pr_auc_per_class
    
    def plot_confidence_distribution(self, confidence, correct):
        """Plot confidence distributions for correct vs incorrect predictions"""
        plt.figure(figsize=(8, 6))
        plt.hist(confidence[correct], bins=20, alpha=0.6, label='Correct', color='green', range=(0, 1))
        plt.hist(confidence[~correct], bins=20, alpha=0.6, label='Incorrect', color='red', range=(0, 1))
        plt.xlabel('Confidence (max softmax probability)')
        plt.ylabel('Count')
        plt.title('Confidence Distribution - Correct vs Incorrect')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Confidence distribution saved as 'confidence_distribution.png'")
    
    def plot_calibration_curve(self, y_true, y_pred, probabilities, n_bins=10):
        """Plot a reliability diagram and compute Expected Calibration Error"""
        confidence = np.max(probabilities, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        n = len(confidence)
        
        bin_confidences = []
        bin_accuracies = []
        ece = 0.0
        
        for i in range(n_bins):
            lower = i / n_bins
            upper = (i + 1) / n_bins
            if i == 0:
                mask = confidence <= upper
            else:
                mask = (confidence > lower) & (confidence <= upper)
            size = mask.sum()
            if size > 0:
                bin_conf = confidence[mask].mean()
                bin_acc = accuracies[mask].mean()
                ece += (size / n) * abs(bin_acc - bin_conf)
            else:
                bin_conf = (lower + upper) / 2
                bin_acc = 0.0
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
        plt.plot(bin_confidences, bin_accuracies, marker='o', label='Model calibration')
        plt.xlabel('Mean Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Calibration Curve - EfficientNetB4 (ECE={ece:.4f})')
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Calibration curve saved as 'calibration_curve.png'")
        return ece
    
    def plot_generalization_gap(self, history):
        """Plot train-vs-val accuracy and loss gaps to visualize overfitting"""
        epochs = range(1, len(history['train_loss']) + 1)
        accuracy_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(epochs, accuracy_gap, 'b-o', label='Accuracy Gap (train - val)')
        ax1.set_title('Accuracy Generalization Gap')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy Gap (%)')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(epochs, loss_gap, 'r-o', label='Loss Gap (val - train)')
        ax2.set_title('Loss Generalization Gap')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Gap')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig('generalization_gap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Generalization gap saved as 'generalization_gap.png'")
    
    def plot_trainable_parameters(self, history):
        """Plot the number of trainable parameters per epoch"""
        epochs = range(1, len(history['trainable_params']) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history['trainable_params'], 'm-o')
        plt.title('Trainable Parameters over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Trainable Parameters')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('trainable_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Trainable parameters saved as 'trainable_parameters.png'")
    
    def train(self):
        """Full training pipeline with gradual unfreezing"""
        train_loader, val_loader, test_loader = self.create_dataloaders()
        model = self.build_model()
        criterion = nn.CrossEntropyLoss()
        
        # Initial optimizer (only head)
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': [],
            'trainable_params': []
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

            # Track trainable parameters for the unfreeze-jump plot
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Validation
            val_loss, val_acc, val_preds, val_targets, _ = self.validate(model, val_loader, criterion, epoch)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['trainable_params'].append(trainable_params)
            
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
        
        # Final evaluation on the best model (held-out test set, touched once)
        final_loss, final_acc, final_preds, final_targets, final_probs = self.validate(model, test_loader, criterion, best_epoch)
        
        # Plot training curves + training diagnostics
        self.plot_training_curves(history)
        self.plot_trainable_parameters(history)
        self.plot_generalization_gap(history)
        
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
        
        # ===== EXTENDED FINAL METRICS =====
        balanced_acc = balanced_accuracy_score(final_targets, final_preds)
        print(f"\nBALANCED ACCURACY: {balanced_acc:.4f}")
        
        macro_precision = precision_score(final_targets, final_preds, average='macro', zero_division=0)
        macro_recall = recall_score(final_targets, final_preds, average='macro', zero_division=0)
        macro_f1 = f1_score(final_targets, final_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(final_targets, final_preds, average='weighted', zero_division=0)
        
        # Per-class specificity, ROC-AUC, PR-AUC
        cm = confusion_matrix(final_targets, final_preds, labels=np.arange(self.num_classes))
        specificity_per_class = {}
        roc_auc_per_class = {}
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            specificity_per_class[class_name] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            binary_targets = (final_targets == i).astype(int)
            roc_auc_per_class[class_name] = roc_auc_score(binary_targets, final_probs[:, i])
        
        macro_roc_auc = roc_auc_score(final_targets, final_probs, multi_class='ovr', average='macro')
        weighted_roc_auc = roc_auc_score(final_targets, final_probs, multi_class='ovr', average='weighted')
        print(f"MACRO ROC-AUC: {macro_roc_auc:.4f}")
        print(f"WEIGHTED ROC-AUC: {weighted_roc_auc:.4f}")
        
        self.plot_roc_curves(final_targets, final_probs)
        pr_auc_per_class = self.plot_pr_curves(final_targets, final_probs)
        macro_pr_auc = np.mean(list(pr_auc_per_class.values()))
        print(f"MACRO PR-AUC: {macro_pr_auc:.4f}")
        
        # Top-2 / Top-3 accuracy
        top2_preds = np.argsort(final_probs, axis=1)[:, -2:]
        top3_preds = np.argsort(final_probs, axis=1)[:, -3:]
        top2_acc = np.mean([t in p for t, p in zip(final_targets, top2_preds)])
        top3_acc = np.mean([t in p for t, p in zip(final_targets, top3_preds)])
        print(f"\nTOP-1 ACCURACY: {final_acc:.2f}%")
        print(f"TOP-2 ACCURACY: {top2_acc * 100:.2f}%")
        print(f"TOP-3 ACCURACY: {top3_acc * 100:.2f}%")
        
        # Confidence analysis
        confidence = np.max(final_probs, axis=1)
        correct = final_preds == final_targets
        mean_confidence = np.mean(confidence)
        correct_confidence = np.mean(confidence[correct]) if np.any(correct) else 0.0
        incorrect_confidence = np.mean(confidence[~correct]) if np.any(~correct) else 0.0
        print(f"\nMean confidence: {mean_confidence:.4f}")
        print(f"Mean confidence when correct: {correct_confidence:.4f}")
        print(f"Mean confidence when incorrect: {incorrect_confidence:.4f}")
        
        self.plot_confusion_matrix(final_targets, final_preds, best_epoch)
        self.plot_final_confusion_matrix(final_targets, final_preds)
        self.plot_confidence_distribution(confidence, correct)
        ece = self.plot_calibration_curve(final_targets, final_preds, final_probs)
        print(f"EXPECTED CALIBRATION ERROR: {ece:.4f}")
        
        # ===== FINAL MODEL EVALUATION SUMMARY =====
        print("\n" + "="*60)
        print("FINAL MODEL EVALUATION")
        print("="*60)
        print(f"\nOverall Accuracy       : {final_acc / 100:.4f}")
        print(f"Balanced Accuracy      : {balanced_acc:.4f}")
        print(f"\nMacro Precision        : {macro_precision:.4f}")
        print(f"Macro Recall           : {macro_recall:.4f}")
        print(f"Macro F1               : {macro_f1:.4f}")
        print(f"\nWeighted F1            : {weighted_f1:.4f}")
        print(f"\nMacro ROC-AUC          : {macro_roc_auc:.4f}")
        print(f"Weighted ROC-AUC       : {weighted_roc_auc:.4f}")
        print(f"\nMacro PR-AUC           : {macro_pr_auc:.4f}")
        print(f"\nTop-1 Accuracy         : {final_acc / 100:.4f}")
        print(f"Top-2 Accuracy         : {top2_acc:.4f}")
        print(f"Top-3 Accuracy         : {top3_acc:.4f}")
        print(f"\nMean Confidence        : {mean_confidence:.4f}")
        print(f"Correct Confidence     : {correct_confidence:.4f}")
        print(f"Incorrect Confidence   : {incorrect_confidence:.4f}")
        print(f"\nExpected Calibration Error : {ece:.4f}")
        
        print("\n------------------------------------------------------------")
        print("PER-CLASS METRICS")
        print("------------------------------------------------------------")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'Specificity':<12} "
              f"{'F1':<12} {'ROC-AUC':<10} {'PR-AUC':<10}")
        print("-"*72)
        for class_name in self.class_names:
            m = report[class_name]
            print(f"{class_name:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                  f"{specificity_per_class[class_name]:<12.4f} {m['f1-score']:<12.4f} "
                  f"{roc_auc_per_class[class_name]:<10.4f} {pr_auc_per_class[class_name]:<10.4f}")
        
        # Save metrics
        metrics = {
            'final_accuracy': final_acc,
            'best_accuracy': best_val_acc,
            'total_training_time': total_training_time,
            'epochs_trained': len(history['train_loss']),
            'best_epoch': best_epoch,
            'history': history,
            'balanced_accuracy': balanced_acc,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'macro_roc_auc': macro_roc_auc,
            'weighted_roc_auc': weighted_roc_auc,
            'macro_pr_auc': macro_pr_auc,
            'top2_accuracy': top2_acc,
            'top3_accuracy': top3_acc,
            'mean_confidence': mean_confidence,
            'correct_confidence': correct_confidence,
            'incorrect_confidence': incorrect_confidence,
            'ece': ece,
            'specificity_per_class': specificity_per_class,
            'roc_auc_per_class': roc_auc_per_class,
            'pr_auc_per_class': pr_auc_per_class
        }
        torch.save(metrics, 'efficientnetb4_training_metrics.pth')
        
        return model, history, metrics

if __name__ == "__main__":
    trainer = EfficientNetB4GradualUnfreezingTrainer(
        data_dir='HAM10000_split/train',
        batch_size=64,
        num_epochs=50,
        learning_rate=0.001,
        sampler_mode='sqrt'
    )
    model, history, metrics = trainer.train()
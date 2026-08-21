import torch
import torch.nn as nn
import torch.nn.functional as F
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
import glob
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed):
    """Seed every RNG the pipeline touches so fresh runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), computed from log_softmax output.

    alpha: optional per-class weights (tensor on the model device); None = unweighted.
    TODO: label smoothing could be folded into the log-prob term if needed.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = target_log_probs.exp()
        loss = -(1.0 - p_t) ** self.gamma * target_log_probs
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class EfficientNetB4GradualUnfreezingTrainer:
    def __init__(self, data_dir, val_data_dir='HAM10000_split/val', test_data_dir='HAM10000_split/test', batch_size=None, num_epochs=50, learning_rate=0.001, image_size=380, seed=42, imbalance_strategy='focal', sampler_mode='sqrt', loss_mode='weighted_ce', focal_gamma=2.0, grad_accum_steps=1, weight_decay=1e-4, label_smoothing=0.05, resume_checkpoint=None, early_stop_patience=7, stage2_unfreeze_patience=4, use_tta=True, use_ensemble=True, keep_top_k_checkpoints=3):
        """
        EfficientNetB4 with gradual unfreezing:
        - Epochs 1-4: Train only head (frozen backbone)
        - Epoch 5: Unfreeze blocks 6, 7, 8 (deepest blocks)
        - Blocks 4-5: unfrozen only after stage2_unfreeze_patience epochs without a val
          Macro-F1 improvement (plateau-gated, not a fixed epoch), with BatchNorm running
          stats pinned to eval mode for those blocks (small-batch stabilizer)
        - data_dir = raw TRAIN split (HAM10000_split/train); augmentation is on-the-fly
        - val_data_dir = raw VAL split, used only for best-model / early-stop / LR step
        - test_data_dir = raw TEST split, evaluated exactly once at the end
        - image_size = train/eval resolution; EfficientNetB4's native input is 380x380 (default)
        - seed = RNG seed for a fresh (non-resume) run; resume restores saved RNG state instead
        - imbalance_strategy = single imbalance-correction mechanism:
            'sampler_only' = WeightedRandomSampler + plain CE
            'loss_only'    = plain shuffle + weighted CE (1/sqrt(count), mean-normalized)
            'focal'        = plain shuffle + FocalLoss (default)
            'both'         = WeightedRandomSampler + weighted CE (old stacked behavior, for comparison)
        - sampler_mode = 'sqrt' (1/sqrt(count), default) | 'inverse' (1/count) | 'none' (plain shuffle)
        - loss_mode = 'ce' (baseline) | 'weighted_ce' (1/sqrt(count) weights, mean-normalized to ~1)
        - focal_gamma = focusing parameter for FocalLoss (default 2.0)
        - grad_accum_steps = accumulate gradients over this many batches before stepping, so
          effective batch size = batch_size * grad_accum_steps (default 1 = no accumulation)
        - weight_decay = AdamW weight decay, applied to every param group (incl. unfreezes)
        - label_smoothing = CE smoothing (focal loss skips it; see FocalLoss TODO)
        - resume_checkpoint = path to efficientnetb4_last_checkpoint.pth to resume from, or None for a fresh run
        - early_stop_patience = epochs without validation Macro-F1 improvement before stopping
        - stage2_unfreeze_patience = epochs without val Macro-F1 improvement that must pass
          before blocks 4-5 unfreeze fires (plateau-gated; default 4, fires at most once)
        - use_tta = average softmax over augmented views (flip/±5° rotation) for the final
          test-set evaluation, and print the no-TTA → TTA delta (default True)
        - use_ensemble = after training, average the top-K best checkpoints' softmax
          probabilities on the test set and report ensemble metrics (default True)
        - keep_top_k_checkpoints = how many best checkpoints to keep for the ensemble
          (default 3); the single best is always also saved to
          efficientnetb4_best_model.pth for resume/tooling compatibility
        """
        if loss_mode not in ('ce', 'weighted_ce'):
            raise ValueError(
                f"Unknown loss_mode: {loss_mode!r}. Supported values: 'ce', 'weighted_ce'."
            )
        if imbalance_strategy not in ('sampler_only', 'loss_only', 'focal', 'both'):
            raise ValueError(
                f"Unknown imbalance_strategy: {imbalance_strategy!r}. Supported values: "
                f"'sampler_only', 'loss_only', 'focal', 'both'."
            )
        if batch_size is None:
            # 380px tensors are ~2.9x the pixels of 224px; batch 16 (vs 32) keeps memory in
            # range on a ~6 GB GPU. grad_accum_steps restores the effective batch size if needed.
            batch_size = 16 if image_size >= 320 else 32
        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.seed = seed
        self.imbalance_strategy = imbalance_strategy
        self.sampler_mode = sampler_mode
        self.loss_mode = loss_mode
        self.focal_gamma = focal_gamma
        self.grad_accum_steps = grad_accum_steps
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.resume_checkpoint = resume_checkpoint
        self.early_stop_patience = early_stop_patience
        if stage2_unfreeze_patience < 1:
            raise ValueError(
                f"stage2_unfreeze_patience must be >= 1, got {stage2_unfreeze_patience}."
            )
        if keep_top_k_checkpoints < 1:
            raise ValueError(
                f"keep_top_k_checkpoints must be >= 1, got {keep_top_k_checkpoints}."
            )
        self.stage2_unfreeze_patience = stage2_unfreeze_patience
        self.use_tta = use_tta
        self.use_ensemble = use_ensemble
        self.keep_top_k_checkpoints = keep_top_k_checkpoints
        self._stage2_unfrozen = False
        self._stage2_epoch = None
        self._stage2_best_epoch = None
        self._frozen_bn_modules = []
        self._best_checkpoints = []
        self.num_classes = 7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['NV', 'MEL', 'BKL', 'BCC', 'AKIEC', 'VASC', 'DF']

        print(f"Using device: {self.device}")
        print("="*60)
        print("EfficientNetB4 with Gradual Unfreezing (Blocks 6-8 @ epoch 5; blocks 4-5 plateau-gated)")
        print("="*60)
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Random seed: {self.seed}")
        print(f"Imbalance strategy: {self.imbalance_strategy}")
        print(f"Loss mode: {self.loss_mode}")
        print(f"Sampler mode: {self.sampler_mode}")
        print(f"Model selection: Macro-F1")
        print(f"Blocks 4-5 unfreezing: plateau-gated (patience {self.stage2_unfreeze_patience}) with BN running stats frozen")
        print(f"Resume: {self.resume_checkpoint}")
        print("="*60)

    def _worker_init_fn(self, worker_id):
        """Seed numpy/random per worker from torch's initial_seed() (itself derived from
        the loader generator's manual_seed), so worker-side transform randomness and
        sampler draws are reproducible across runs with the same seed."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def create_dataloaders(self):
        """Train on the raw train split with on-the-fly augmentation; early-stop on raw val; final eval on raw test."""
        # EfficientNetB4 native input size is 380x380; self.image_size (default 380) keeps
        # fine dermoscopic detail (pigment network, borders, blue-white veil) that 224px loses
        # Train: on-the-fly augmentation, conservative geometric only (no vertical flip, no black fill)
        resize_scale = int(self.image_size * 256 / 224)
        train_transform = transforms.Compose([
            transforms.Resize((resize_scale, resize_scale)),
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
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

        # Single imbalance-correction mechanism per run: sampler only for
        # 'sampler_only'/'both', plain shuffle otherwise.
        use_sampler = self.imbalance_strategy in ('sampler_only', 'both')
        class_counts = np.bincount(train_dataset.targets, minlength=self.num_classes)
        self.class_counts = class_counts
        if use_sampler and self.sampler_mode == 'sqrt':
            class_weights = 1.0 / np.sqrt(class_counts)
        elif use_sampler and self.sampler_mode == 'inverse':
            class_weights = 1.0 / class_counts
        else:
            class_weights = None
            if use_sampler:
                raise ValueError(
                    f"Unknown sampler_mode: {self.sampler_mode}. 'none' (plain shuffle) is "
                    f"contradictory with imbalance_strategy={self.imbalance_strategy!r}; use "
                    f"'loss_only' or 'focal' for plain-shuffle training."
                )

        train_generator = torch.Generator().manual_seed(self.seed)
        val_generator = torch.Generator().manual_seed(self.seed)
        test_generator = torch.Generator().manual_seed(self.seed)

        if class_weights is not None:
            sample_weights = np.array([class_weights[t] for t in train_dataset.targets])
            train_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(train_dataset), replacement=True,
                generator=train_generator
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, generator=train_generator, worker_init_fn=self._worker_init_fn)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, generator=train_generator, worker_init_fn=self._worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, generator=val_generator, worker_init_fn=self._worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, generator=test_generator, worker_init_fn=self._worker_init_fn)

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
        # TODO: metadata fusion — HAM10000 metadata (age/sex/localization) could be fused via a
        # small MLP branch concatenated with pooled features before this Linear, gated behind a
        # use_metadata flag; skipped because the ImageFolder pipeline has no metadata plumbing.
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

    def unfreeze_blocks(self, model, block_indices, optimizer, lr_factor=0.1, freeze_bn=False):
        """
        Unfreeze specific EfficientNet blocks and add to optimizer
        block_indices: list of block numbers to unfreeze (e.g., [29, 30, 31])
        freeze_bn: also pin BatchNorm running stats in those blocks to eval mode —
        affine weight/bias stays trainable (requires_grad=True as set by the param
        loop) but running_mean/running_var stop updating. Those modules are recorded
        in self._frozen_bn_modules so train_epoch() can re-apply eval mode after
        every model.train() call.
        """
        parameters_to_add = []
        unfrozen_count = 0
        block_prefixes = [f'features.{idx}.' for idx in block_indices]  # EfficientNet blocks are in features.{index}

        for name, param in model.named_parameters():
            # Check if parameter belongs to any of the specified blocks
            for prefix in block_prefixes:
                if prefix in name:
                    param.requires_grad = True
                    parameters_to_add.append(param)
                    unfrozen_count += param.numel()
                    break

        if freeze_bn:
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d) and any(name.startswith(p) for p in block_prefixes):
                    module.eval()
                    if module not in self._frozen_bn_modules:
                        self._frozen_bn_modules.append(module)
            if self._frozen_bn_modules:
                print(f"   → BatchNorm running stats frozen in blocks {block_indices} "
                      f"(eval mode; affine params still trainable)")

        if parameters_to_add:
            optimizer.add_param_group({
                'params': parameters_to_add,
                'lr': self.learning_rate * lr_factor,
                'weight_decay': self.weight_decay
            })
            print(f"   → Unfrozen {unfrozen_count:,} parameters in blocks {block_indices}")
            return unfrozen_count
        return 0

    def _build_criterion(self):
        """
        Dispatch on the single active imbalance strategy:
        - 'sampler_only': plain CrossEntropyLoss (sampler does the rebalancing)
        - 'loss_only': CrossEntropyLoss with mild 1/sqrt(class_count) weights,
          mean-normalized to ~1.0 (never full inverse-frequency, which over-corrects)
        - 'focal': FocalLoss(gamma=self.focal_gamma); alpha = 1/sqrt(count)
          mean-normalized weights when loss_mode='weighted_ce', else unweighted
        - 'both': weighted CrossEntropyLoss on top of the sampler (old stacked
          behavior, kept for comparison)
        """
        counts = self.class_counts.astype(np.float64)
        class_weights = 1.0 / np.sqrt(counts)
        class_weights = class_weights / class_weights.mean()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        if self.imbalance_strategy == 'focal':
            alpha = self.class_weights if self.loss_mode == 'weighted_ce' else None
            return FocalLoss(gamma=self.focal_gamma, alpha=alpha)
        if self.imbalance_strategy == 'loss_only':
            if self.loss_mode == 'ce':
                return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            return nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=self.label_smoothing)
        if self.imbalance_strategy == 'both':
            return nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=self.label_smoothing)
        # 'sampler_only'
        return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        # Blocks unfrozen with freeze_bn=True must never flip back to train mode:
        # their BatchNorm running stats are pinned by keeping those modules in eval().
        for bn in self._frozen_bn_modules:
            bn.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", leave=False, ncols=100)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            output = model(data)
            # Scale loss by 1/grad_accum_steps so accumulated gradients average to the
            # per-batch loss scale; optimizer.step() only fires every grad_accum_steps batches.
            loss = criterion(output, target) / self.grad_accum_steps
            loss.backward()
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})

        # Flush gradients from a trailing partial accumulation window
        if len(train_loader) % self.grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

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

    def predict_with_tta(self, model, loader, n_augments=4):
        """Test-time augmentation for the final test-set evaluation: run each image
        through n_augments deterministic views (identity, horizontal flip, ±5°
        rotation, optionally a 0.95-scale center crop) and average the softmax
        probabilities (not logits) across views. Returns (avg_probs, targets).

        Per-epoch val checks keep using validate() — TTA is final-eval-only here.
        """
        model.eval()
        n_augments = max(1, min(n_augments, 5))
        all_avg_probs = []
        all_targets = []
        crop_size = int(self.image_size * 0.95)
        crop_offset = (self.image_size - crop_size) // 2

        pbar = tqdm(loader, desc="Test (TTA)", leave=False, ncols=100)
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                probs_sum = None
                for aug_idx in range(n_augments):
                    variant = data
                    if aug_idx == 1:
                        variant = torch.flip(data, dims=[3])
                    elif aug_idx == 2:
                        variant = transforms.functional.rotate(data, 5)
                    elif aug_idx == 3:
                        variant = transforms.functional.rotate(data, -5)
                    elif aug_idx == 4:
                        variant = transforms.functional.resized_crop(
                            data, crop_offset, crop_offset, crop_size, crop_size,
                            (self.image_size, self.image_size))
                    probs = torch.softmax(model(variant), dim=1)
                    probs_sum = probs if probs_sum is None else probs_sum + probs
                all_avg_probs.append((probs_sum / n_augments).cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        return np.concatenate(all_avg_probs, axis=0), np.array(all_targets)

    def _trim_checkpoints(self):
        """Keep only the top-K best checkpoints (by val Macro-F1); delete worse ones
        from disk so the rotation never grows unbounded."""
        keep = max(1, self.keep_top_k_checkpoints)
        entries = sorted(self._best_checkpoints, key=lambda e: e['f1'], reverse=True)
        for entry in entries[keep:]:
            if os.path.exists(entry['path']):
                os.remove(entry['path'])
                print(f"   → Removed {entry['path']} (outside top-{keep})")
        self._best_checkpoints = entries[:keep]

    def _rescan_checkpoints(self):
        """Rebuild the top-K list from the epoch-named best checkpoints found on disk
        (resume path: the in-memory list is not persisted in old checkpoints)."""
        entries = []
        for path in sorted(glob.glob('efficientnetb4_best_model_epoch*.pth')):
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
            except Exception:
                continue
            entries.append({
                'path': path,
                'f1': ckpt.get('best_macro_f1', -1.0),
                'epoch': ckpt.get('epoch', -1),
            })
        entries.sort(key=lambda e: e['f1'], reverse=True)
        self._best_checkpoints = entries[:max(1, self.keep_top_k_checkpoints)]

    def evaluate_ensemble(self, test_loader, criterion=None):
        """Load the top-K saved best checkpoints, run each through predict_with_tta
        (or plain validate when use_tta=False) on the test set, average their softmax
        probabilities across checkpoints, and report ensemble Macro-F1 / balanced
        accuracy alongside the single-best-model numbers."""
        paths = [e['path'] for e in self._best_checkpoints]
        if not paths:
            print("\nNo top-K best checkpoints available — skipping ensemble evaluation.")
            return None

        print("\n" + "=" * 60)
        print("ENSEMBLE EVALUATION (top-K best checkpoints)")
        print("=" * 60)
        model = self.build_model()
        probs_sum = None
        targets = None
        loaded = 0
        for i, entry in enumerate(self._best_checkpoints, start=1):
            path = entry['path']
            if not os.path.exists(path):
                continue
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            member_epoch = checkpoint['epoch']
            member_f1 = checkpoint['best_macro_f1']
            if self.use_tta:
                probs, targets = self.predict_with_tta(model, test_loader, n_augments=4)
            else:
                _, _, _, targets, probs = self.validate(model, test_loader, criterion, member_epoch - 1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            loaded += 1
            print(f"  member {i}/{len(paths)}: {path} (val Macro-F1 {member_f1:.4f} @ epoch {member_epoch})")

        if probs_sum is None:
            print("  No ensemble members could be loaded — skipping.")
            return None

        avg_probs = probs_sum / loaded
        preds = avg_probs.argmax(axis=1)
        ens_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        ens_bacc = balanced_accuracy_score(targets, preds)
        print(f"\n  ENSEMBLE Macro-F1: {ens_f1:.4f} | Balanced accuracy: {ens_bacc:.4f} "
              f"({loaded} members)")
        print("=" * 60)
        return {'macro_f1': ens_f1, 'balanced_accuracy': ens_bacc, 'paths': paths}

    def plot_training_curves(self, history):
        """Plot and save training curves"""
        epochs = range(1, len(history['train_loss']) + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Training Accuracy
        ax1.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax1.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax1.axvline(x=6, linestyle='--', color='gray', label='Unfreeze Blocks 6-8')
        ax1.set_title('Training & Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)

        # Training Loss
        ax2.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax2.axvline(x=6, linestyle='--', color='gray', label='Unfreeze Blocks 6-8')
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
        """Plot confusion matrix for an epoch (epoch is 1-based)"""
        self._plot_confusion_matrix(
            y_true, y_pred,
            f'Confusion Matrix - EfficientNetB4 (Epoch {epoch})',
            f'efficientnetb4_confusion_matrix_epoch_{epoch}.png'
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
        """Full training pipeline: macro-F1 model selection, resume support, per-epoch last checkpoint."""
        # Fresh-start reproducibility: seed every RNG before dataloaders/models are built.
        # (The resume path below restores saved RNG state instead.)
        set_seed(self.seed)

        if self.resume_checkpoint is None:
            # Fresh run: clear stale epoch-named best checkpoints from any previous
            # experiment so the top-K rotation/ensemble only ever covers this run.
            for stale in glob.glob('efficientnetb4_best_model_epoch*.pth'):
                os.remove(stale)
                print(f"   → Removed stale best checkpoint from a previous run: {stale}")
            self._best_checkpoints = []

        train_loader, val_loader, test_loader = self.create_dataloaders()
        model = self.build_model()
        criterion = self._build_criterion()

        # ===== EXPERIMENT CONFIGURATION =====
        print("\n" + "=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)
        print(f"Model: EfficientNet-B4")
        print(f"Dataset: HAM10000")
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Random seed: {self.seed}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"\nImbalance strategy: {self.imbalance_strategy} (single mechanism)")
        if self.imbalance_strategy == 'sampler_only':
            print(f"    Sampler: {self.sampler_mode} WeightedRandomSampler; loss: plain CE (label_smoothing={self.label_smoothing})")
        elif self.imbalance_strategy == 'loss_only':
            print(f"    Sampler: none (plain shuffle); loss: weighted CE (1/sqrt(count), mean-normalized)")
        elif self.imbalance_strategy == 'focal':
            alpha_desc = "weighted alpha (1/sqrt(count), mean-normalized)" if self.loss_mode == 'weighted_ce' else "no per-class alpha"
            print(f"    Sampler: none (plain shuffle); loss: FocalLoss(gamma={self.focal_gamma}) with {alpha_desc}")
        else:
            print(f"    Sampler: {self.sampler_mode} WeightedRandomSampler + weighted CE (stacked, comparison only)")
        print(f"Model selection: Validation Macro-F1")
        print(f"Maximum epochs: {self.num_epochs}")
        print(f"Early stopping patience: {self.early_stop_patience}")
        print(f"\nGradual unfreezing:")
        print(f"    Head: epochs 1-4")
        print(f"    Blocks 6-8: epoch 5+")
        print(f"    Blocks 4-5: plateau-gated — fires once after {self.stage2_unfreeze_patience} epochs "
              f"without a val Macro-F1 improvement (BatchNorm running stats frozen)")
        print(f"Final test evaluation: TTA {'on' if self.use_tta else 'off'} | "
              f"Top-{self.keep_top_k_checkpoints} ensemble {'on' if self.use_ensemble else 'off'}")
        print(f"\nCheckpoint:")
        print(f"    Best: efficientnetb4_best_model.pth")
        print(f"    Resume: efficientnetb4_last_checkpoint.pth")
        print("=" * 60)

        if hasattr(self, 'class_weights'):
            print(f"\nClass weights / focal alpha (1/sqrt(count), mean-normalized to ~1.0):")
            for name, w in zip(self.class_names, self.class_weights.cpu().numpy()):
                print(f"    {name}: {w:.4f}")

        # Initial optimizer (only head)
        optimizer = optim.AdamW(model.classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Scheduler tracks the same metric used for model selection: val Macro-F1 (mode='max').
        # val_loss is dominated by the majority class on this imbalanced dataset and can diverge.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
        )

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_macro_f1': [],
            'lr': [],
            'trainable_params': []
        }

        best_macro_f1 = -1.0
        best_val_acc = 0.0
        best_epoch = None
        patience_counter = 0

        start_epoch = 0
        if self.resume_checkpoint is None:
            print("\nNo checkpoint specified → starting from epoch 1")
        else:
            if not os.path.exists(self.resume_checkpoint):
                raise FileNotFoundError(
                    f"Cannot resume training: checkpoint file not found: {self.resume_checkpoint}"
                )
            try:
                # Load on CPU: RNG state tensors must stay CPU ByteTensors for set_rng_state;
                # load_state_dict copies model/optimizer state onto the target device itself.
                checkpoint = torch.load(self.resume_checkpoint, map_location='cpu', weights_only=False)
            except Exception as e:
                print("=" * 60)
                print("RESUME FAILED: could not load the checkpoint.")
                print(f"Checkpoint: {self.resume_checkpoint}")
                print(f"Error: {e}")
                print("Training will NOT start from epoch 1. Fix the checkpoint and try again.")
                print("=" * 60)
                raise

            ckpt_loss = checkpoint.get('loss_mode')
            ckpt_sampler = checkpoint.get('sampler_mode')
            ckpt_strategy = checkpoint.get('imbalance_strategy')
            if (ckpt_loss != self.loss_mode or ckpt_sampler != self.sampler_mode
                    or ckpt_strategy != self.imbalance_strategy):
                raise ValueError(
                    f"Cannot resume: checkpoint was created with loss_mode={ckpt_loss!r}, "
                    f"sampler_mode={ckpt_sampler!r}, imbalance_strategy={ckpt_strategy!r}, but this "
                    f"run uses loss_mode={self.loss_mode!r}, sampler_mode={self.sampler_mode!r}, "
                    f"imbalance_strategy={self.imbalance_strategy!r}. Refusing to silently produce a "
                    f"misleading experiment. Delete/rename the checkpoint to start fresh."
                )

            if checkpoint['epoch'] >= 6:
                # Blocks 6-8 were already unfrozen when this checkpoint was saved; re-apply the
                # unfreeze so the optimizer param-group layout matches before state restore.
                self.unfreeze_blocks(model, [6, 7, 8], optimizer, lr_factor=0.1)
            # Stage-2 state: old checkpoints (pre-plateau-gating) fired at saved epoch > 25;
            # new checkpoints persist the flag directly.
            self._stage2_unfrozen = checkpoint.get('stage2_unfrozen', checkpoint['epoch'] > 25)
            self._stage2_epoch = checkpoint.get('stage2_epoch')
            self._stage2_best_epoch = checkpoint.get('stage2_best_epoch')
            self._best_checkpoints = checkpoint.get('best_checkpoints', [])
            if self._stage2_unfrozen:
                # Re-apply so the optimizer param-group layout (and frozen-BN eval mode)
                # matches before state restore.
                self.unfreeze_blocks(model, [4, 5], optimizer, lr_factor=0.1, freeze_bn=True)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_macro_f1 = checkpoint['best_metric']
            best_val_acc = checkpoint.get('best_val_accuracy', 0.0)
            best_epoch = checkpoint['best_epoch']
            patience_counter = checkpoint['patience_counter']
            history = checkpoint['history']

            torch.set_rng_state(checkpoint['rng_state'])
            if checkpoint.get('cuda_rng_state') is not None and torch.cuda.is_available():
                cuda_states = checkpoint['cuda_rng_state']
                try:
                    # torch >= 2.13 keeps CUDA RNG state as CPU byte tensors
                    torch.cuda.set_rng_state_all(cuda_states)
                except TypeError:
                    # older torch expects CUDA byte tensors
                    torch.cuda.set_rng_state_all([s.cuda() for s in cuda_states])
            if checkpoint.get('numpy_rng_state') is not None:
                np.random.set_state(checkpoint['numpy_rng_state'])
            if checkpoint.get('python_rng_state') is not None:
                random.setstate(checkpoint['python_rng_state'])

            start_epoch = checkpoint['epoch']  # next epoch to run (0-indexed)

            # Rebuild the top-K list from disk: the in-memory list is not persisted
            # in older checkpoints and files may have been trimmed externally.
            self._rescan_checkpoints()

            print("=" * 60)
            print("RESUMING TRAINING")
            print("=" * 60)
            print(f"Checkpoint: {self.resume_checkpoint}")
            print(f"Resuming from epoch: {start_epoch + 1}")
            print(f"Previous best validation metric: {best_macro_f1:.4f}")
            print(f"Previous best epoch: {best_epoch}")
            print("=" * 60)

        if start_epoch >= self.num_epochs:
            print(f"Checkpoint already reached epoch {self.num_epochs}; no epochs left to train.")

        print("\n" + "=" * 60)
        print("Training with Gradual Unfreezing")
        print("=" * 60)
        print(f"Epochs 1-4:   Train head only")
        print(f"Epochs 5+:    Train head + blocks 6-8 (unfrozen at epoch 5)")
        print(f"Blocks 4-5:   Train head + blocks 4-8 once {self.stage2_unfreeze_patience} epochs pass "
              f"without a val Macro-F1 improvement (BN running stats frozen; batch size {self.batch_size})")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(start_epoch, self.num_epochs):
            # ===== GRADUAL UNFREEZING LOGIC =====
            if epoch == 5:
                print(f"\n🔓 UNFREEZING blocks 6-8 at epoch {epoch+1}...")
                # EfficientNetB4 has blocks 0-8 (features.0 to features.8)
                # Unfreeze blocks 6, 7, 8 (deepest blocks)
                self.unfreeze_blocks(model, [6, 7, 8], optimizer, lr_factor=0.1)

            if (not self._stage2_unfrozen and epoch > 5
                    and patience_counter >= self.stage2_unfreeze_patience):
                # Blocks 4-5 unfreeze is plateau-gated, not epoch-gated: it fires once,
                # only after blocks 6-8 have had room to help, and only once val Macro-F1
                # has gone stage2_unfreeze_patience epochs without improving. BatchNorm
                # running stats inside these blocks are pinned to eval mode (small-batch
                # stabilizer) — affine params stay trainable.
                print(f"\n🔓 UNFREEZING blocks 4-5 at epoch {epoch+1} "
                      f"(no val Macro-F1 improvement for {patience_counter} epochs)...")
                self._stage2_unfrozen = True
                self._stage2_epoch = epoch + 1
                self._stage2_best_epoch = best_epoch
                self.unfreeze_blocks(model, [4, 5], optimizer, lr_factor=0.1, freeze_bn=True)

            # Training
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)

            # Track trainable parameters for the unfreeze-jump plot
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Validation
            val_loss, val_acc, val_preds, val_targets, _ = self.validate(model, val_loader, criterion, epoch)

            # Macro-F1 is the model-selection metric (HAM10000 is highly imbalanced)
            val_macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_macro_f1'].append(val_macro_f1)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['trainable_params'].append(trainable_params)

            # Scheduler step (tracks val Macro-F1, the same metric used for model selection)
            scheduler.step(val_macro_f1)

            # ===== ENHANCED EPOCH SUMMARY =====
            print("\n" + "=" * 80)
            print(f"📊 EPOCH {epoch+1}/{self.num_epochs} SUMMARY")
            print("=" * 80)
            print(f"🎯 Training   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
            print(f"✅ Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}% | Macro F1: {val_macro_f1:.4f}")
            print(f"📈 Best Val Macro F1: {best_macro_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            # ===== BEST-MODEL UPDATE (only when validation Macro-F1 improves) =====
            if val_macro_f1 > best_macro_f1:
                improvement = val_macro_f1 - best_macro_f1
                best_macro_f1 = val_macro_f1
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0

                best_state = {
                    'model_state_dict': model.state_dict(),
                    'epoch': best_epoch,
                    'best_macro_f1': best_macro_f1,
                    'best_val_accuracy': best_val_acc,
                    'best_val_loss': val_loss,
                }
                # Rotating top-K filenames: keep the best K by Macro-F1 on disk for the
                # ensemble, delete worse ones. The canonical unqualified filename is kept
                # as a copy of the single best so resume/external tooling is unaffected.
                rotating_path = f'efficientnetb4_best_model_epoch{best_epoch}.pth'
                torch.save(best_state, rotating_path)
                self._best_checkpoints = [e for e in self._best_checkpoints if e['path'] != rotating_path]
                self._best_checkpoints.append({'path': rotating_path, 'f1': best_macro_f1, 'epoch': best_epoch})
                self._trim_checkpoints()
                torch.save(best_state, 'efficientnetb4_best_model.pth')

                print(f"💾 SAVING MODEL → Epoch {epoch+1} | New Best Macro F1: {best_macro_f1:.4f} (+{improvement:.4f}) | Acc: {best_val_acc:.2f}%")
                self.plot_confusion_matrix(val_targets, val_preds, epoch + 1)

            else:
                patience_counter += 1
                print(f"⏳ NO IMPROVEMENT → Epoch {epoch+1} | Patience: {patience_counter}/{self.early_stop_patience}")

            # ===== LAST CHECKPOINT (end of a fully completed epoch; overwritten every epoch) =====
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_macro_f1,
                'best_epoch': best_epoch,
                'best_val_accuracy': best_val_acc,
                'patience_counter': patience_counter,
                'history': history,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
                'loss_mode': self.loss_mode,
                'sampler_mode': self.sampler_mode,
                'imbalance_strategy': self.imbalance_strategy,
                'stage2_unfrozen': self._stage2_unfrozen,
                'stage2_epoch': self._stage2_epoch,
                'stage2_best_epoch': self._stage2_best_epoch,
                'best_checkpoints': self._best_checkpoints,
            }, 'efficientnetb4_last_checkpoint.pth')

            # ===== TRAINING METRICS (updated at the end of every completed epoch) =====
            torch.save({
                'history': history,
                'best_macro_f1': best_macro_f1,
                'best_epoch': best_epoch,
                'current_epoch': epoch + 1,
                'loss_mode': self.loss_mode,
                'sampler_mode': self.sampler_mode,
                'imbalance_strategy': self.imbalance_strategy,
            }, 'efficientnetb4_training_metrics.pth')

            # ===== EARLY STOPPING (monitors validation Macro-F1) =====
            if patience_counter >= self.early_stop_patience:
                print(f"\n🛑 EARLY STOPPING triggered at epoch {epoch+1}")
                break

        total_training_time = time.time() - start_time

        # ===== STAGE-2 ROLLBACK NOTE (informational only) =====
        if (self._stage2_unfrozen and self._stage2_epoch is not None
                and best_epoch is not None and best_epoch < self._stage2_epoch):
            print(f"\nℹ️ STAGE-2 NOTE: the blocks 4-5 unfreeze (epoch {self._stage2_epoch}) did not help "
                  f"this run — the best epoch ({best_epoch}) predates it. The best-checkpoint "
                  f"mechanism keeps the saved model unaffected.")

        # ===== FINAL RESULTS =====
        print("\n" + "=" * 60)
        print("🏁 TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total time: {total_training_time:.2f} seconds")
        print(f"Best Validation Macro F1: {best_macro_f1:.4f} (epoch {best_epoch})")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

        # Evaluate the BEST Macro-F1 model on the held-out test set (touched exactly once)
        best_model_path = 'efficientnetb4_best_model.pth'
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"{best_model_path} not found — cannot evaluate the best model.")
        best_checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        best_epoch = best_checkpoint['epoch']

        # Run the no-TTA baseline first so the TTA delta is visible, not silently swapped in.
        final_loss, no_tta_acc, no_tta_preds, no_tta_targets, no_tta_probs = self.validate(
            model, test_loader, criterion, best_epoch - 1)
        no_tta_f1 = f1_score(no_tta_preds, no_tta_targets, average='macro', zero_division=0)

        if self.use_tta:
            final_probs, final_targets = self.predict_with_tta(model, test_loader, n_augments=4)
            final_preds = final_probs.argmax(axis=1)
            final_acc = 100.0 * float((final_preds == final_targets).mean())
            tta_f1 = f1_score(final_preds, final_targets, average='macro', zero_division=0)
            print(f"\nTest Macro-F1: no-TTA {no_tta_f1:.4f} → TTA {tta_f1:.4f}")
        else:
            final_preds, final_targets, final_probs = no_tta_preds, no_tta_targets, no_tta_probs
            final_acc = no_tta_acc
            tta_f1 = no_tta_f1

        # Plot training curves + training diagnostics
        self.plot_training_curves(history)
        self.plot_trainable_parameters(history)
        self.plot_generalization_gap(history)

        # Print final metrics
        print("\n" + "=" * 60)
        print("📋 FINAL CLASSIFICATION REPORT")
        print("=" * 60)
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
        print("\n" + "=" * 60)
        print("FINAL MODEL EVALUATION")
        print("=" * 60)
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

        # ===== ENSEMBLE EVALUATION (top-K best checkpoints) =====
        ensemble_result = None
        if self.use_ensemble:
            ensemble_result = self.evaluate_ensemble(test_loader, criterion)

        # ===== EXACT CONFIGURATION USED (reproducibility recap) =====
        print("\n" + "=" * 60)
        print("EXPERIMENT CONFIGURATION (as run)")
        print("=" * 60)
        print(f"Model: EfficientNet-B4 (ImageNet pretrained)")
        print(f"Dataset: HAM10000 | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Batch size: {self.batch_size} | Grad accum: {self.grad_accum_steps} (effective batch {self.batch_size * self.grad_accum_steps}) | Learning rate: {self.learning_rate} | Max epochs: {self.num_epochs}")
        print(f"Optimizer: AdamW (weight_decay={self.weight_decay}) | Scheduler: ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7) on val Macro-F1 (mode='max')")
        print(f"Imbalance strategy: {self.imbalance_strategy} (loss_mode={self.loss_mode}, sampler_mode={self.sampler_mode}) | Label smoothing: {self.label_smoothing}")
        print(f"Unfreezing: head (epochs 1-4) → +blocks 6-8 (epoch 5+) → +blocks 4-5 "
              f"(plateau-gated: {self.stage2_unfreeze_patience} epochs without val Macro-F1 "
              f"improvement; BN running stats frozen)")
        print(f"Model selection: Validation Macro-F1 | Early stop patience: {self.early_stop_patience}")
        print(f"Random seed: {self.seed}")
        print(f"Class order: {self.class_names}")
        print(f"Best validation Macro F1: {best_macro_f1:.4f} @ epoch {best_epoch}")
        print("=" * 60)

        # ===== STAGE COMPARISON (what actually moved the needle) =====
        print("\n" + "=" * 60)
        print("STAGE COMPARISON")
        print("=" * 60)
        stage2_triggered = self._stage2_unfrozen and self._stage2_epoch is not None
        if stage2_triggered:
            pre_f1s = history['val_macro_f1'][:self._stage2_epoch - 1]
            post_f1s = history['val_macro_f1'][self._stage2_epoch - 1:]
            if pre_f1s:
                pre_best_epoch = int(np.argmax(pre_f1s)) + 1
                pre_best_f1 = float(np.max(pre_f1s))
            else:
                pre_best_epoch, pre_best_f1 = 'n/a', 'n/a'
            if post_f1s:
                post_best_epoch = int(np.argmax(post_f1s)) + self._stage2_epoch
                post_best_f1 = float(np.max(post_f1s))
            else:
                post_best_epoch, post_best_f1 = 'n/a', 'n/a'
        else:
            pre_best_epoch, pre_best_f1 = best_epoch, best_macro_f1
            post_best_epoch, post_best_f1 = 'not triggered', 'n/a'
        pre_f1_str = f"{pre_best_f1:.4f}" if isinstance(pre_best_f1, float) else str(pre_best_f1)
        post_f1_str = f"{post_best_f1:.4f}" if isinstance(post_best_f1, float) else str(post_best_f1)
        tta_note = "" if self.use_tta else " (TTA off — identical by design)"
        print(f"  Best epoch (head+blocks6-8 only): {pre_best_epoch}, Macro-F1 {pre_f1_str}")
        print(f"  Best epoch (+blocks4-5): {post_best_epoch}, Macro-F1 {post_f1_str}")
        print(f"  TTA delta: {no_tta_f1:.4f} -> {tta_f1:.4f}{tta_note}")
        if ensemble_result is not None:
            print(f"  Ensemble (top-K) Macro-F1: {ensemble_result['macro_f1']:.4f}")
        else:
            print(f"  Ensemble (top-K) Macro-F1: n/a")
        print("=" * 60)

        # ===== SAVE FULL METRICS =====
        metrics = {
            # experiment identity (requirements 12 & 18)
            'loss_mode': self.loss_mode,
            'sampler_mode': self.sampler_mode,
            'imbalance_strategy': self.imbalance_strategy,
            'image_size': self.image_size,
            'seed': self.seed,
            'grad_accum_steps': self.grad_accum_steps,
            'weight_decay': self.weight_decay,
            'label_smoothing': self.label_smoothing,
            'num_epochs': self.num_epochs,
            'early_stop_patience': self.early_stop_patience,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            # history & selection
            'history': history,
            'best_macro_f1': best_macro_f1,
            'best_epoch': best_epoch,
            'current_epoch': len(history['train_loss']),
            'best_val_accuracy': best_val_acc,
            # final test-set evaluation
            'final_accuracy': final_acc,
            'total_training_time': total_training_time,
            'epochs_trained': len(history['train_loss']),
            'balanced_accuracy': balanced_acc,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_roc_auc': macro_roc_auc,
            'weighted_roc_auc': weighted_roc_auc,
            'macro_pr_auc': macro_pr_auc,
            'top1_accuracy': final_acc,
            'top2_accuracy': top2_acc,
            'top3_accuracy': top3_acc,
            'mean_confidence': mean_confidence,
            'correct_confidence': correct_confidence,
            'incorrect_confidence': incorrect_confidence,
            'ece': ece,
            'specificity_per_class': specificity_per_class,
            'roc_auc_per_class': roc_auc_per_class,
            'pr_auc_per_class': pr_auc_per_class,
            'classification_report': report,
            # stage-2 unfreeze / TTA / ensemble results
            'stage2_unfrozen': self._stage2_unfrozen,
            'stage2_epoch': self._stage2_epoch,
            'stage2_unfreeze_patience': self.stage2_unfreeze_patience,
            'no_tta_macro_f1': no_tta_f1,
            'tta_macro_f1': tta_f1,
            'use_tta': self.use_tta,
            'ensemble_macro_f1': ensemble_result['macro_f1'] if ensemble_result else None,
            'use_ensemble': self.use_ensemble,
            'keep_top_k_checkpoints': self.keep_top_k_checkpoints,
        }
        torch.save(metrics, 'efficientnetb4_training_metrics.pth')

        return model, history, metrics

if __name__ == "__main__":
    # Auto-resume when a last checkpoint exists (crash recovery); start fresh otherwise.
    resume_path = 'efficientnetb4_last_checkpoint.pth'
    if not os.path.exists(resume_path):
        resume_path = None
    trainer = EfficientNetB4GradualUnfreezingTrainer(
        data_dir='HAM10000_split/train',
        num_epochs=50,
        learning_rate=0.001,
        resume_checkpoint=resume_path
    )
    model, history, metrics = trainer.train()
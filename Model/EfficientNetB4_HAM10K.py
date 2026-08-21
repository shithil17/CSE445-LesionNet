"""
EXP-B4-01-NORM-FINETUNE — EfficientNet-B4 fine-tuning experiment on HAM10000.

The training recipe is the KNOWN-GOOD recipe (val Macro-F1 ~0.773, test Macro-F1
(TTA) ~0.750) restored as the shared baseline for the image-only vs image+metadata
comparison:
  - Correct ImageNet normalization: mean/std come from the torchvision B4 weights
    metadata (``EfficientNet_B4_Weights.IMAGENET1K_V1.transforms()``), applied
    identically to train / validation / test.
  - Aspect-ratio-preserving resize + crop instead of blindly stretching to 380x380:
      train: Resize(shorter side) -> RandomResizedCrop(380) -> geometric aug ->
             color aug -> ToTensor -> Normalize
      val/test: Resize(shorter side) -> CenterCrop(380) -> ToTensor -> Normalize
  - Discriminative learning rates: head 1e-3, blocks 6-8 = head * 0.1 (1e-4),
    blocks 4-5 = head * 0.05 (5e-5) — the LRs that worked in the 0.66 -> 0.77
    journey (defaults derived from lr_head, never hardcoded).
  - ReduceLROnPlateau (mode='max', factor=0.5, patience=5, min_lr=1e-7) on the
    validation Macro-F1 — the scheduler that produced the known-good results.
  - Gradual unfreezing: head only (epochs 1-5), + blocks 6-8 fixed at epoch 6
    (consistently worked), blocks 4-5 plateau-gated (fires once after
    stage2_unfreeze_patience epochs without a val Macro-F1 improvement, with
    BatchNorm running stats in those blocks frozen to eval mode). Blocks 0-3 stay
    frozen for the whole experiment.
  - AMP (fp16 autocast + GradScaler) for the RTX 4050 laptop GPU.
  - Conservative augmentation with the correct ordering (crop/geometric -> color ->
    ToTensor -> Normalize -> optional mild RandomErasing).
  - Single imbalance mechanism: plain shuffle + FocalLoss with mild 1/sqrt(count)
    alpha weighting. CE / weighted-CE remain available for later ablations via
    loss_mode; the WeightedRandomSampler path was removed.
  - Checkpoints: best by validation Macro-F1 (primary) AND best by validation
    accuracy; top-K rotating best checkpoints for the F1-weighted ensemble.
  - TTA only for the final test evaluation. Temperature scaling (Guo et al.,
    post-hoc T fit on val logits) and the F1-weighted top-K ensemble (val-gated
    usage decision) are ON by default — `use_ensemble=False` opt-outs exist.
  - Opt-in Phase-5 improvements: use_mixup / use_cutmix (soft-target focal),
    use_multiscale_tta (second-scale TTA at tta_second_scale px).

Metadata fusion experiment (EXP-B4-META-01, opt-in via use_metadata=True):
  - Leakage-safe preprocessing of HAM10000 metadata (age / sex / localization)
    fitted on the TRAINING split only; validation/test are transformed with the
    fitted state and can never alter it.
  - Image branch stays EfficientNet-B4; a small metadata MLP
    (Linear(meta_dim -> 64) -> ReLU -> Dropout(0.10) -> Linear(64 -> 32)) produces a
    32-d embedding that is concatenated with the pooled image features before the
    classifier (input 1792 + 32). TTA reuses the metadata embedding across image views.
  - use_metadata=False keeps the exact image-only baseline (no metadata dependency
    anywhere in the model path).

Class index order = ImageFolder alphabetical order: ['AKIEC', 'BCC', 'BKL', 'DF',
'MEL', 'NV', 'VASC'] — read from the dataset folders, never hardcoded (the training
docstring order is NOT the index order).

The lesion-level train/val/test split (seed 42; lesions never cross split
boundaries) is defined by split_ham10000.py and is NOT touched here — this script
only consumes HAM10000_split/{train,val,test}.

Run from the Model/ directory:
    ../.venv/bin/python EfficientNetB4_HAM10K.py                     # image-only baseline
    ../.venv/bin/python EfficientNetB4_HAM10K.py --use-metadata      # image + metadata
"""

import csv
import glob
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

EXPERIMENT_ID = "EXP-B4-01-NORM-FINETUNE"    # image-only baseline (default)
METADATA_EXPERIMENT_ID = "EXP-B4-META-01"    # image + metadata fusion (opt-in)
DEFAULT_METADATA_CSV = "../Dataset/HAM10000_groundtruth.csv"

# ImageNet statistics associated with the B4 weights themselves — not hardcoded.
_B4_WEIGHTS = models.EfficientNet_B4_Weights.IMAGENET1K_V1
_WEIGHTS_TRANSFORMS = _B4_WEIGHTS.transforms()
IMAGENET_MEAN = list(_WEIGHTS_TRANSFORMS.mean)
IMAGENET_STD = list(_WEIGHTS_TRANSFORMS.std)


def set_seed(seed):
    """Seed every RNG the pipeline touches so fresh runs are reproducible.
    CuDNN determinism is an explicit choice (logged at startup): it costs a little
    throughput versus benchmark mode, but keeps training bit-reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), computed from log_softmax.

    alpha: optional per-class weights (tensor on the model device); None = unweighted.

    targets may be hard labels (int64 1-D) or soft one-hot mixtures (float 2-D,
    produced by MixUp/CutMix). For soft targets the focal modulating factor is
    applied per class with the model's own p_k (the Menghini et al.
    multi-label-consistent form) and alpha is weighted by the DOMINANT label of
    each mixed pair — a deliberate simplification of the full
    multi-label-consistent extension (noted in the code; see forward()).
    """

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        if targets.ndim == 1:
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            p_t = target_log_probs.exp()
            loss = -(1.0 - p_t) ** self.gamma * target_log_probs
            if self.alpha is not None:
                loss = self.alpha[targets] * loss
        else:
            # Soft targets (MixUp/CutMix): interpolate the one-hot targets the
            # same way the images were mixed. Simplification: the focal
            # modulating factor is applied per class with p_k (multi-label-
            # consistent form) but the per-class alpha weight comes from the
            # DOMINANT label in each mixed pair (not the full Menghini et al.
            # extension) — kept deliberately simple.
            loss = -((1.0 - probs) ** self.gamma * log_probs * targets).sum(dim=1)
            if self.alpha is not None:
                loss = self.alpha[targets.argmax(dim=1)] * loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class HAM10000MetadataPreprocessor:
    """Leakage-safe metadata encoder for EXP-B4-META-01 (age / sex / localization).

    fit() MUST be called with TRAINING metadata records only — it computes the age
    median/mean/std and the sex + localization vocabularies. transform() only
    READS the fitted state and can never alter or refit it (verified at startup).
    Validation and test records are transformed with the train-fitted state.

    Missing-vs-category semantics (GT-03-METADATA-CORRECTION):
      * A missing source value has a *_missing flag of 1 and yields an ALL-ZERO
        one-hot for its category block — it is NOT an artificial category.
      * 'unknown' is a genuine non-missing category in HAM10000_metadata.csv
        (57 sex / 234 localization rows) and is learned as a category like any
        other — it is never a synthetic fallback.
      * A validation/test category that did not occur in TRAIN is 'unknown to
        the encoder': all-zero one-hot with the *_missing flag PRESERVED from
        the source record (unseen != missing).

    Determinism: category vocabularies are the SORTED set of non-missing TRAINING
    values; feature ordering is fixed. Feature vector (float32), in fixed order:
        age_scaled, age_missing,
        sex_<train cat...>, sex_missing,
        localization_<train cat...>, localization_missing

    State is plain JSON-serializable data (no fitted sklearn/torch objects) so it
    can be audited, saved standalone, and embedded in checkpoints. The state
    carries metadata_schema_version; incompatible older states are REFUSED on
    load (stale artifacts must never be reused with a new schema).
    """

    METADATA_SCHEMA_VERSION = 3
    EPSILON = 1e-6

    def __init__(self):
        self.age_median = None
        self.age_mean = None
        self.age_std = None
        self.sex_categories = []
        self.localization_categories = []
        self.feature_names = []
        self.dim = 0

    # ------------------------------------------------------------------ fitting

    def fit(self, records):
        """Fit on TRAINING records only. records: list of dicts with the CSV keys
        age, age_missing, sex, sex_missing, localization, localization_missing."""
        ages = np.array([self._parse_age(r["age"]) for r in records], dtype=np.float64)
        if np.isfinite(np.nanmedian(ages)):
            self.age_median = float(np.nanmedian(ages))
        else:
            self.age_median = 0.0  # degenerate: every training age missing

        imputed = np.where(np.isnan(ages), self.age_median, ages)
        mean = float(np.mean(imputed))
        std = float(np.std(imputed))
        if std < self.EPSILON:
            std = 1.0
        self.age_mean = mean
        self.age_std = std

        # Sorted sets of non-missing TRAINING category values -> deterministic.
        # 'unknown' is included only if it is a real non-missing source category
        # in the training split; it is never manufactured.
        self.sex_categories = sorted({
            r["sex"].strip().lower()
            for r in records
            if not self._as_int(r["sex_missing"]) and r["sex"].strip()
        })
        self.localization_categories = sorted({
            r["localization"].strip().lower()
            for r in records
            if not self._as_int(r["localization_missing"]) and r["localization"].strip()
        })

        self.feature_names = (
            ["age_scaled", "age_missing"]
            + [f"sex_{c}" for c in self.sex_categories]
            + ["sex_missing"]
            + [f"localization_{c}" for c in self.localization_categories]
            + ["localization_missing"]
        )
        self.dim = len(self.feature_names)
        return self

    # ------------------------------------------------------------------ transform

    def transform(self, records):
        """Transform arbitrary records with the fitted state (pure read-only).
        Returns a float32 (N, dim) array."""
        if self.age_mean is None:
            raise RuntimeError("HAM10000MetadataPreprocessor.fit() must run before transform()")
        sex_index = {c: i for i, c in enumerate(self.sex_categories)}
        loc_index = {c: i for i, c in enumerate(self.localization_categories)}

        out = np.zeros((len(records), self.dim), dtype=np.float32)
        for i, r in enumerate(records):
            col = 0
            age = self._parse_age(r["age"])
            age_missing = self._as_int(r["age_missing"])
            if age_missing or np.isnan(age):
                age = self.age_median
            out[i, col] = (age - self.age_mean) / self.age_std
            col += 1
            out[i, col] = float(age_missing)
            col += 1

            sex = r["sex"].strip().lower()
            sex_missing = self._as_int(r["sex_missing"])
            sex_idx = sex_index.get(sex)
            if sex_idx is not None:
                out[i, col + sex_idx] = 1.0
            # Missing sex -> all-zero one-hot + sex_missing=1.
            # Unseen-but-valid sex -> all-zero one-hot + sex_missing preserved.
            col += len(self.sex_categories)
            out[i, col] = float(sex_missing)
            col += 1

            loc = r["localization"].strip().lower()
            loc_missing = self._as_int(r["localization_missing"])
            loc_idx = loc_index.get(loc)
            if loc and not loc_missing and loc_idx is not None:
                out[i, col + loc_idx] = 1.0
            # Missing localization -> all-zero one-hot + localization_missing=1.
            # Unseen-but-valid localization -> all-zero one-hot + flag preserved.
            col += len(self.localization_categories)
            out[i, col] = float(loc_missing)
        return out

    def fit_transform(self, records):
        return self.fit(records).transform(records)

    # ------------------------------------------------------------------ state

    def state_dict(self):
        return {
            "metadata_schema_version": self.METADATA_SCHEMA_VERSION,
            "age_median": self.age_median,
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "sex_categories": self.sex_categories,
            "localization_categories": self.localization_categories,
            "feature_names": self.feature_names,
            "dim": self.dim,
        }

    def load_state_dict(self, state):
        if state.get("metadata_schema_version") != self.METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"Metadata preprocessor state schema "
                f"v{state.get('metadata_schema_version')} is incompatible with "
                f"current schema v{self.METADATA_SCHEMA_VERSION}. Delete the stale "
                f"metadata_preprocessor_state.json / checkpoint and regenerate."
            )
        for key in ("age_median", "age_mean", "age_std", "sex_categories",
                    "localization_categories", "feature_names", "dim"):
            if key not in state:
                raise ValueError(f"metadata preprocessor state missing key: {key}")
            setattr(self, key, state[key])
        return self

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            return self.load_state_dict(json.load(f))

    @staticmethod
    def _parse_age(raw):
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            return float("nan")
        return value if value >= 0 else float("nan")

    @staticmethod
    def _as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1 if str(value).strip() not in ("", "0") else 0


class HAM10000Dataset(Dataset):
    """ImageFolder with an optional per-image metadata tensor.

    use_metadata=True: returns (image, metadata_tensor_float32, target)
    use_metadata=False: returns (image, target) — identical to plain ImageFolder.

    The metadata tensors are precomputed once (fit on train only, done by the
    trainer BEFORE loader creation) and indexed by dataset position, so no
    preprocessing ever happens inside worker processes."""
    def __init__(self, root, transform, metadata_tensors=None):
        self.image_folder = datasets.ImageFolder(root=root, transform=transform)
        self.metadata_tensors = metadata_tensors
        if metadata_tensors is not None and len(metadata_tensors) != len(self.image_folder):
            raise ValueError(
                f"metadata tensor count ({len(metadata_tensors)}) != image count "
                f"({len(self.image_folder)}) for {root}"
            )

    @property
    def classes(self):
        return self.image_folder.classes

    @property
    def targets(self):
        return self.image_folder.targets

    @property
    def samples(self):
        return self.image_folder.samples

    @property
    def image_ids(self):
        return [os.path.splitext(os.path.basename(p))[0] for p, _ in self.image_folder.samples]

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, target = self.image_folder[idx]
        if self.metadata_tensors is None:
            return image, target
        return image, self.metadata_tensors[idx].float(), target


class EfficientNetB4FinetuneTrainer:
    """EfficientNet-B4 with correct ImageNet preprocessing, aspect-preserving
    geometry, discriminative LRs and gradual unfreezing (restored known-good recipe).

    Key arguments (everything else has a sensible default for this experiment):
      - data_dir / val_data_dir / test_data_dir: the lesion-level split folders
      - image_size=380: model input resolution (kept, do not increase on a 4050)
      - eval_resize=380: shorter-side resize for val/test before CenterCrop
      - batch_size=16, grad_accum_steps=1 (fallback 8 + 2 keeps effective 16)
      - loss_mode: 'focal' (default) | 'weighted_ce' | 'ce'  (imbalance ablation)
      - lr_head=1e-3: peak head LR (the value that worked in the 0.66 -> 0.77
        journey). lr_blocks_6_8 / lr_blocks_4_5 default to lr_head * 0.1 /
        lr_head * 0.05 (never hardcoded); pass explicit values to override.
      - scheduler: ReduceLROnPlateau(mode='max', factor=0.5, patience=5,
        min_lr=1e-7) on the validation Macro-F1, stepped at the end of each
        validation phase (no warmup/cosine — that experiment is closed).
      - Unfreezing: head (epochs 1-5), + blocks 6-8 fixed at epoch 6, blocks 4-5
        plateau-gated via stage2_unfreeze_patience (default 4) epochs without a
        val Macro-F1 improvement, fires once, BatchNorm running stats in blocks
        4-5 frozen to eval mode on firing (affine params stay trainable).
      - use_amp=True: fp16 autocast + GradScaler
      - use_tta=True: TTA only for the final test evaluation
      - use_ensemble=True: F1-weighted averaging of the top-K best checkpoints;
        used on the test set only if it beats the single best model on VALIDATION
        (the decision never looks at test data). keep_top_k_checkpoints (default 3).
      - resume_checkpoint: efficientnetb4_last_checkpoint.pth path or None
      - use_metadata=False: image-only baseline (EXP-B4-01-NORM-FINETUNE).
        True -> image + metadata fusion (EXP-B4-META-01): HAM10000 metadata is
        preprocessed leakage-safely (train-only fit) and fused via a small MLP.
      - use_mixup / use_cutmix: opt-in Phase-5 augmentation (soft-target focal;
        both on -> one picked randomly per batch)
      - use_multiscale_tta: opt-in Phase-5 second-scale TTA (tta_second_scale px)
    """

    def __init__(
        self,
        data_dir,
        val_data_dir='HAM10000_split/val',
        test_data_dir='HAM10000_split/test',
        experiment_id=None,
        batch_size=16,
        grad_accum_steps=1,
        num_epochs=50,
        image_size=380,
        eval_resize=380,
        seed=42,
        loss_mode='focal',
        focal_gamma=2.0,
        weight_decay=1e-4,
        lr_head=1e-3,
        lr_blocks_6_8=None,
        lr_blocks_4_5=None,
        stage2_unfreeze_patience=4,
        use_amp=True,
        use_random_erasing=False,
        resume_checkpoint=None,
        early_stop_patience=7,
        use_tta=True,
        use_ensemble=True,
        keep_top_k_checkpoints=3,
        use_metadata=False,
        metadata_csv=DEFAULT_METADATA_CSV,
        use_mixup=False,
        use_cutmix=False,
        use_multiscale_tta=False,
        tta_second_scale=456,
    ):
        if loss_mode not in ('ce', 'weighted_ce', 'focal'):
            raise ValueError(
                f"Unknown loss_mode: {loss_mode!r}. Supported: 'ce', 'weighted_ce', 'focal'."
            )
        if batch_size < 1 or grad_accum_steps < 1:
            raise ValueError("batch_size and grad_accum_steps must be >= 1.")
        if eval_resize < image_size:
            raise ValueError(
                f"eval_resize ({eval_resize}) must be >= image_size ({image_size}) "
                f"so CenterCrop never fails."
            )
        if stage2_unfreeze_patience < 1:
            raise ValueError(
                f"stage2_unfreeze_patience must be >= 1, got {stage2_unfreeze_patience}."
            )
        if keep_top_k_checkpoints < 1:
            raise ValueError(
                f"keep_top_k_checkpoints must be >= 1, got {keep_top_k_checkpoints}."
            )
        if use_mixup and use_cutmix:
            print("NOTE: both use_mixup and use_cutmix are enabled — one is picked "
                  "randomly per batch (standard practice).")

        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.use_metadata = use_metadata
        self.metadata_csv = metadata_csv
        self.experiment_id = experiment_id or (
            METADATA_EXPERIMENT_ID if use_metadata else EXPERIMENT_ID
        )
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.eval_resize = eval_resize
        self.seed = seed
        self.loss_mode = loss_mode
        self.focal_gamma = focal_gamma
        self.weight_decay = weight_decay
        self.lr_head = lr_head
        self.lr_blocks_6_8 = lr_blocks_6_8 if lr_blocks_6_8 is not None else lr_head * 0.1
        self.lr_blocks_4_5 = lr_blocks_4_5 if lr_blocks_4_5 is not None else lr_head * 0.05
        self.stage2_unfreeze_patience = stage2_unfreeze_patience
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_random_erasing = use_random_erasing
        self.resume_checkpoint = resume_checkpoint
        self.early_stop_patience = early_stop_patience
        self.use_tta = use_tta
        self.use_ensemble = use_ensemble
        self.keep_top_k_checkpoints = keep_top_k_checkpoints
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.use_multiscale_tta = use_multiscale_tta
        self.tta_second_scale = tta_second_scale

        self._stage2_unfrozen = False
        self._stage2_epoch = None
        self._frozen_bn_modules = []
        self._best_checkpoints = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.num_classes = 7  # overwritten from the dataset folders in create_dataloaders
        self.class_names = None
        self.class_counts = None
        self.class_weights = None
        self.preprocessor = None       # HAM10000MetadataPreprocessor (metadata mode)
        self.metadata_dim = 0
        self._fresh_preprocessor_state = None  # JSON snapshot for leak/resume checks

    # ------------------------------------------------------------------ data

    def _worker_init_fn(self, worker_id):
        """Seed numpy/random per worker from torch's initial_seed() (itself derived
        from the loader generator's manual_seed) so worker-side transform randomness
        is reproducible across runs with the same seed."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _build_transforms(self):
        """Deterministic transform recipes (order matters):

        train: aspect-preserving Resize(shorter side) -> RandomResizedCrop ->
               geometric aug (flips, mild rotation) -> color aug (mild ColorJitter) ->
               ToTensor -> ImageNet Normalize -> optional mild RandomErasing
        val/test: aspect-preserving Resize(shorter side) -> CenterCrop ->
                  ToTensor -> ImageNet Normalize  (no random augmentation)

        The initial shorter-side resize keeps RandomResizedCrop crops at or below
        native resolution (no upsampling) and preserves the original aspect ratio —
        a rectangular dermoscopic image is never stretched into 380x380."""
        resize_short = int(self.image_size * 256 / 224)  # 434 for 380 (ImageNet-style)
        train_transform = transforms.Compose([
            transforms.Resize(resize_short),
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        if self.use_random_erasing:
            train_transform.transforms.append(
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
            )

        eval_transform = transforms.Compose([
            transforms.Resize(self.eval_resize),  # int -> shorter side, aspect preserved
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return train_transform, eval_transform

    def create_dataloaders(self):
        """Plain-shuffle train loader + single-pass val/test loaders on the raw split.
        Metadata mode (use_metadata=True): metadata preprocessing is fitted on the
        TRAINING split only and applied to train/val/test BEFORE loaders are built —
        never inside worker processes."""
        train_transform, eval_transform = self._build_transforms()

        if self.use_metadata:
            metadata_tensors = self._prepare_metadata()
            train_dataset = HAM10000Dataset(
                root=self.data_dir, transform=train_transform,
                metadata_tensors=metadata_tensors["train"])
            val_dataset = HAM10000Dataset(
                root=self.val_data_dir, transform=eval_transform,
                metadata_tensors=metadata_tensors["val"])
            test_dataset = HAM10000Dataset(
                root=self.test_data_dir, transform=eval_transform,
                metadata_tensors=metadata_tensors["test"])
        else:
            train_dataset = datasets.ImageFolder(root=self.data_dir, transform=train_transform)
            val_dataset = datasets.ImageFolder(root=self.val_data_dir, transform=eval_transform)
            test_dataset = datasets.ImageFolder(root=self.test_data_dir, transform=eval_transform)

        # Class order = ImageFolder alphabetical order (single source of truth).
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)

        for name, ds in [("val", val_dataset), ("test", test_dataset)]:
            if self.class_names != ds.classes:
                raise ValueError(
                    f"Train/{name} class folders differ: {self.class_names} vs {ds.classes}"
                )

        self.class_counts = np.bincount(train_dataset.targets, minlength=self.num_classes)

        train_generator = torch.Generator().manual_seed(self.seed)
        val_generator = torch.Generator().manual_seed(self.seed)
        test_generator = torch.Generator().manual_seed(self.seed)

        # No sampler in this experiment: plain shuffle + focal alpha weighting only.
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, generator=train_generator,
            worker_init_fn=self._worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, generator=val_generator,
            worker_init_fn=self._worker_init_fn,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, generator=test_generator,
            worker_init_fn=self._worker_init_fn,
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Classes (ImageFolder order): {self.class_names}")
        print(f"Train class counts: {dict(zip(self.class_names, self.class_counts))}")
        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------ metadata

    def _load_metadata_records(self):
        """Build metadata_by_image[image_id] = record from the ground-truth CSV.
        Failures: missing file/columns, empty/duplicate image ids. This mapping is
        the ONLY join between images and metadata — never row ordering."""
        if not os.path.exists(self.metadata_csv):
            raise FileNotFoundError(
                f"Metadata CSV not found: {self.metadata_csv} (run build_groundtruth.py first)"
            )
        with open(self.metadata_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)
        required = ["image", "age", "age_missing", "sex", "sex_missing",
                    "localization", "localization_missing"]
        missing_cols = [c for c in required if c not in header]
        if missing_cols:
            raise ValueError(f"Metadata CSV missing required column(s): {missing_cols}")

        metadata_by_image = {}
        for r in rows:
            image = r["image"].strip()
            if not image:
                raise ValueError("Metadata CSV contains an empty 'image' value")
            if image in metadata_by_image:
                raise ValueError(f"Metadata CSV contains duplicate image: {image}")
            metadata_by_image[image] = {k: r[k] for k in required}
        return metadata_by_image

    def _prepare_metadata(self):
        """Fit the preprocessor on TRAIN metadata only, transform all three splits
        with that fitted state, run the startup leakage checks, and persist the
        standalone artifact. Returns {'train': t, 'val': t, 'test': t} float32
        tensors. Called once, before DataLoader creation."""
        metadata_by_image = self._load_metadata_records()

        records = {}
        for name in ("train", "val", "test"):
            root = self.data_dir if name == "train" else getattr(self, f"{name}_data_dir")
            imgs = [os.path.splitext(os.path.basename(p))[0]
                    for p, _ in datasets.ImageFolder(root=root).samples]
            missing = sorted({i for i in imgs if i not in metadata_by_image})
            if missing:
                raise ValueError(
                    f"{name}: {len(missing)}/{len(imgs)} images have no metadata "
                    f"record (e.g. {missing[:5]}) — refusing to train with a "
                    f"partial metadata join"
                )
            records[name] = [metadata_by_image[i] for i in imgs]

        preprocessor = HAM10000MetadataPreprocessor()
        preprocessor.fit(records["train"])          # TRAIN ONLY
        self.preprocessor = preprocessor
        self.metadata_dim = preprocessor.dim
        self._fresh_preprocessor_state = json.dumps(
            preprocessor.state_dict(), sort_keys=True, default=float)

        tensors = {}
        for name in ("train", "val", "test"):
            tensors[name] = torch.from_numpy(preprocessor.transform(records[name])).float()
            # Leakage check: transforming val/test must NEVER alter the fitted state.
            if json.dumps(preprocessor.state_dict(), sort_keys=True, default=float) \
                    != self._fresh_preprocessor_state:
                raise RuntimeError(
                    f"metadata preprocessor state changed after transforming {name} — "
                    f"preprocessing leakage detected"
                )

        preprocessor.save("metadata_preprocessor_state.json")
        print("\n" + "=" * 60)
        print("METADATA PREPROCESSING (EXP-B4-META-01)")
        print("=" * 60)
        print("Metadata preprocessing:")
        print(f"    Age median: {preprocessor.age_median:.4f}")
        print(f"    Age mean: {preprocessor.age_mean:.4f}")
        print(f"    Age std: {preprocessor.age_std:.4f}")
        print(f"    Sex categories: {preprocessor.sex_categories}")
        print(f"    Localization categories ({len(preprocessor.localization_categories)}): "
              f"{preprocessor.localization_categories}")
        print(f"    Final metadata dimension: {preprocessor.dim}")
        print(f"    Feature order: {preprocessor.feature_names}")
        print("Leakage checks:")
        print("    Age statistics fitted on training split only: PASS")
        print("    Categorical vocabularies fitted on training split only: PASS")
        print("    Validation did not alter preprocessing state: PASS")
        print("    Test did not alter preprocessing state: PASS")
        print("    Target labels / image / lesion ids used as metadata: NO")
        print("=" * 60)
        print(f"Metadata feature dimension: {preprocessor.dim}")
        return tensors

    def _preprocessor_state(self):
        return self.preprocessor.state_dict() if self.use_metadata else None

    # ------------------------------------------------------------------ model

    def build_model(self):
        """Image-only path (default): EfficientNetB4 + Dropout(0.2) + Linear(1792, 7)
        — literal mirror of the architecture the deployment checkpoint
        (efficientnetb4_classifier.pth) uses.

        Metadata path (use_metadata=True): same EfficientNet-B4 image branch plus a
        small metadata MLP (Linear(meta_dim -> 64) -> ReLU -> Dropout(0.10) ->
        Linear(64 -> 32)); the 32-d embedding is concatenated with the pooled image
        features, so the final classifier receives 1792 + 32."""
        print("\nBuilding EfficientNetB4 model...")
        model = models.efficientnet_b4(weights=_B4_WEIGHTS)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier[1].in_features
        if self.use_metadata:
            if self.metadata_dim <= 0:
                raise RuntimeError(
                    "metadata_dim is 0 — create_dataloaders() must run before build_model() "
                    "so the train-only preprocessor is fitted first"
                )
            metadata_mlp = nn.Sequential(
                nn.Linear(self.metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, 32),
            )
            model.metadata_mlp = metadata_mlp.to(self.device)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features + 32, self.num_classes),
            )
            print(f"Metadata branch: Linear({self.metadata_dim} -> 64) -> ReLU -> "
                  f"Dropout(0.10) -> Linear(64 -> 32)")
            print(f"Fusion: concatenate image features ({num_features}) + "
                  f"metadata embedding (32) -> classifier input {num_features + 32}")
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, self.num_classes),
            )

        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        return model

    def _forward(self, model, images, metadata=None):
        """Metadata mode: pooled EfficientNet features concatenated with the MLP
        metadata embedding, then the shared classifier. Image-only mode: plain
        model(images) — the exact baseline forward path."""
        if metadata is None:
            return model(images)
        features = model.features(images)
        features = torch.flatten(model.avgpool(features), 1)
        meta_embedding = model.metadata_mlp(metadata)
        combined = torch.cat([features, meta_embedding], dim=1)
        return model.classifier(combined)

    def _features_params(self, model, indices):
        """Parameters of features.{idx}.* for the given block indices."""
        prefixes = [f'features.{idx}.' for idx in indices]
        return [
            p for name, p in model.named_parameters()
            if any(name.startswith(pre) for pre in prefixes)
        ]

    def _set_trainable(self, model, epoch_idx):
        """Apply the current trainable stage for a 0-based epoch index.

        Head (classifier + metadata MLP): always trainable.
        Blocks 6-8: fixed unfreeze at 1-based epoch 6 (0-based idx 5) — this fixed
        epoch consistently worked, so it stays fixed.
        Blocks 4-5: trainable only once the plateau-gated stage-2 unfreeze has
        fired (self._stage2_unfrozen, set in train()); on firing, BatchNorm
        running stats in those blocks are pinned to eval mode (see
        _freeze_bn_stats / train_epoch).
        Blocks 0-3 stay frozen for the entire experiment.

        The optimizer param groups are created once up front; groups whose stage
        has not started simply have requires_grad=False, so AdamW never updates
        them (params without a gradient are skipped). In metadata mode the metadata
        MLP trains from epoch 1 (part of the head group).
        """
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        if self.use_metadata:
            for param in model.metadata_mlp.parameters():
                param.requires_grad = True
        if epoch_idx >= 5:
            for param in self._features_params(model, [6, 7, 8]):
                param.requires_grad = True
        if self._stage2_unfrozen:
            for param in self._features_params(model, [4, 5]):
                param.requires_grad = True

    def _freeze_bn_stats(self, model, block_indices):
        """Pin BatchNorm running stats in the given blocks to permanent eval mode:
        affine weight/bias stay trainable (requires_grad set by _set_trainable),
        but running_mean/running_var stop updating. The modules are recorded in
        self._frozen_bn_modules so train_epoch() re-applies eval mode after every
        model.train() call."""
        prefixes = [f'features.{idx}.' for idx in block_indices]
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and any(
                    name.startswith(p) for p in prefixes):
                module.eval()
                if module not in self._frozen_bn_modules:
                    self._frozen_bn_modules.append(module)
        print(f"   -> BatchNorm running stats frozen in blocks {block_indices} "
              f"(eval mode; affine params still trainable)")

    def _build_param_groups(self, model):
        """Explicit optimizer groups: classifier (+ metadata MLP in metadata mode)
        / blocks 6-8 / blocks 4-5, each with its own LR. Blocks 0-3 are never part
        of any group. Frozen groups receive no gradients (requires_grad=False) and
        are therefore never updated."""
        classifier_params = list(model.classifier.parameters())
        if self.use_metadata:
            classifier_params += list(model.metadata_mlp.parameters())
        return [
            {
                'name': 'classifier' + ('+metadata' if self.use_metadata else ''),
                'params': classifier_params,
                'lr': self.lr_head,
                'weight_decay': self.weight_decay,
            },
            {
                'name': 'blocks 6-8',
                'params': self._features_params(model, [6, 7, 8]),
                'lr': self.lr_blocks_6_8,
                'weight_decay': self.weight_decay,
            },
            {
                'name': 'blocks 4-5',
                'params': self._features_params(model, [4, 5]),
                'lr': self.lr_blocks_4_5,
                'weight_decay': self.weight_decay,
            },
        ]

    def _build_scheduler(self, optimizer):
        """ReduceLROnPlateau on the validation Macro-F1 (mode='max') — the
        scheduler from the known-good recipe. Stepped with scheduler.step(
        val_macro_f1) at the end of each validation phase. The warmup+cosine
        experiment is closed; nothing else is active here."""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7,
        )

    def _build_criterion(self):
        """Single imbalance mechanism per run. The experiment default is plain
        shuffle + FocalLoss with mild 1/sqrt(count) alpha weights (mean-normalized);
        'weighted_ce' and 'ce' remain for later ablation runs."""
        counts = self.class_counts.astype(np.float64)
        class_weights = 1.0 / np.sqrt(counts)
        class_weights = class_weights / class_weights.mean()
        self.class_weights = torch.tensor(
            class_weights, dtype=torch.float32, device=self.device,
        )

        if self.loss_mode == 'focal':
            return FocalLoss(gamma=self.focal_gamma, alpha=self.class_weights)
        if self.loss_mode == 'weighted_ce':
            return nn.CrossEntropyLoss(weight=self.class_weights)
        return nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ train / eval

    @staticmethod
    def _rand_bbox(size, lam):
        """CutMix bounding box (Zhang et al., original paper)."""
        w, h = size[2], size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        return bbx1, bby1, bbx2, bby2

    def _mixup_batch(self, data, target):
        """MixUp (alpha=0.2): convex combination of images AND their one-hot
        targets. Returns (mixed_data, soft_targets)."""
        lam = float(np.random.beta(0.2, 0.2))
        index = torch.randperm(data.size(0), device=data.device)
        mixed = lam * data + (1 - lam) * data[index]
        one_hot = F.one_hot(target, self.num_classes).float()
        soft_targets = lam * one_hot + (1 - lam) * one_hot[index]
        return mixed, soft_targets

    def _cutmix_batch(self, data, target):
        """CutMix (alpha=1.0): replace a bounding box in each image with the
        corresponding patch of a paired image; the one-hot targets are
        interpolated by the patch area fraction. Returns (mixed_data,
        soft_targets)."""
        lam = float(np.random.beta(1.0, 1.0))
        index = torch.randperm(data.size(0), device=data.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(data.shape, lam)
        data = data.clone()
        data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(2) * data.size(3)))
        one_hot = F.one_hot(target, self.num_classes).float()
        soft_targets = lam * one_hot + (1 - lam) * one_hot[index]
        return data, soft_targets

    def train_epoch(self, model, train_loader, criterion, optimizer):
        """One training epoch with AMP (fp16 autocast + GradScaler) and gradient
        accumulation. Loss is scaled by 1/grad_accum_steps; optimizer.step() fires
        every grad_accum_steps batches. Opt-in MixUp/CutMix (use_mixup /
        use_cutmix) is applied per batch during training only — never to
        val/test."""
        model.train()
        # Blocks unfrozen with frozen BN stats must never flip back to train mode:
        # re-apply eval() to those modules right after every model.train() call.
        for bn in self._frozen_bn_modules:
            bn.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        mixed_batches = 0

        pbar = tqdm(train_loader, desc="[Train]", leave=False, ncols=100)
        for batch_idx, batch in enumerate(pbar):
            data, target = batch[0], batch[-1]
            data, target = data.to(self.device), target.to(self.device)
            metadata = batch[1].to(self.device) if self.use_metadata else None

            soft_target = None
            if self.use_mixup or self.use_cutmix:
                # Both enabled -> pick one per batch randomly (standard practice).
                if self.use_mixup and (not self.use_cutmix or random.random() < 0.5):
                    data, soft_target = self._mixup_batch(data, target)
                elif self.use_cutmix:
                    data, soft_target = self._cutmix_batch(data, target)
                mixed_batches += 1

            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                output = self._forward(model, data, metadata)
                loss = criterion(
                    output, soft_target if soft_target is not None else target
                ) / self.grad_accum_steps

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            # Mixed batches: the dominant label of the soft target is the
            # reference for the accuracy display.
            if soft_target is not None:
                correct += (predicted == soft_target.argmax(dim=1)).sum().item()
            else:
                correct += (predicted == target).sum().item()

            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})

        # Flush a trailing partial accumulation window.
        if len(train_loader) % self.grad_accum_steps != 0:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(self, model, val_loader, criterion, epoch_label=""):
        """Single-pass evaluation (no TTA during training)."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []

        pbar = tqdm(val_loader, desc=f"[Val {epoch_label}]", leave=False, ncols=100)
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                data, target = batch[0], batch[-1]
                data, target = data.to(self.device), target.to(self.device)
                metadata = batch[1].to(self.device) if self.use_metadata else None
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                    output = self._forward(model, data, metadata)
                    loss = criterion(output, target)
                    probabilities = torch.softmax(output.float(), dim=1)

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
            np.array(all_probs),
        )

    def predict_with_tta(self, model, loader, n_augments=4, scales=None):
        """Test-time augmentation — final evaluation only, never per-epoch.
        Runs n_augments deterministic views (identity, horizontal flip, ±5°
        rotation, optional 0.95-scale center crop) per input scale, averages the
        LOGITS across views/scales and applies softmax once. scales=None uses
        [self.image_size]; with use_multiscale_tta the second scale
        (tta_second_scale px, e.g. 456) is appended so softmax probabilities are
        averaged across both scales and the augment views.

        Returns (avg_probs, targets, avg_logits) — avg_logits are the
        TTA-averaged logits (temperature scaling fits/consumes these, keeping
        the shipped pipeline consistent).

        Metadata mode: the metadata embedding is computed ONCE per batch and reused
        across all augmented image views (mathematically equivalent to the full
        forward per view, since the metadata branch is view-independent)."""
        model.eval()
        n_augments = max(1, min(n_augments, 5))
        if scales is None:
            scales = [self.image_size]
            if self.use_multiscale_tta:
                scales = sorted(set(scales + [self.tta_second_scale]))
        all_avg_probs = []
        all_targets = []
        all_avg_logits = []
        pbar = tqdm(loader, desc="[Test TTA]", leave=False, ncols=100)
        with torch.no_grad():
            for batch in pbar:
                data, target = batch[0], batch[-1]
                data, target = data.to(self.device), target.to(self.device)
                metadata = batch[1].to(self.device) if self.use_metadata else None
                if self.use_metadata:
                    meta_embedding = model.metadata_mlp(metadata)  # once per sample
                logits_sum = None
                n_views = 0
                for scale in scales:
                    # Keep the 0.95 center crop relative to each input scale.
                    # At the default scale this is unchanged; at the opt-in
                    # second scale it avoids accidentally using a much tighter crop.
                    crop_size = int(scale * 0.95)
                    crop_offset = (scale - crop_size) // 2
                    scaled = data
                    if scale != self.image_size:
                        scaled = F.interpolate(
                            scaled, size=(scale, scale), mode='bilinear',
                            align_corners=False)
                    for aug_idx in range(n_augments):
                        variant = scaled
                        if aug_idx == 1:
                            variant = torch.flip(scaled, dims=[3])
                        elif aug_idx == 2:
                            variant = transforms.functional.rotate(scaled, 5)
                        elif aug_idx == 3:
                            variant = transforms.functional.rotate(scaled, -5)
                        elif aug_idx == 4:
                            variant = transforms.functional.resized_crop(
                                scaled, crop_offset, crop_offset, crop_size, crop_size,
                                (scale, scale),
                            )
                        with torch.autocast("cuda", dtype=torch.float16,
                                            enabled=self.use_amp):
                            if self.use_metadata:
                                features = model.features(variant)
                                features = torch.flatten(model.avgpool(features), 1)
                                logits = model.classifier(
                                    torch.cat([features, meta_embedding], dim=1))
                            else:
                                logits = model(variant)
                        logits_sum = logits if logits_sum is None else logits_sum + logits
                        n_views += 1
                avg_logits = logits_sum / n_views
                all_avg_probs.append(torch.softmax(avg_logits.float(), dim=1).cpu().numpy())
                all_avg_logits.append(avg_logits.float().cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        return (
            np.concatenate(all_avg_probs, axis=0),
            np.array(all_targets),
            np.concatenate(all_avg_logits, axis=0),
        )

    def collect_logits(self, model, loader, epoch_label=""):
        """Single-pass logits + targets (no TTA, no softmax) on a loader — used
        for the no-TTA temperature scaling path and ensemble member evaluation."""
        model.eval()
        all_logits = []
        all_targets = []
        pbar = tqdm(loader, desc=f"[Logits {epoch_label}]", leave=False, ncols=100)
        with torch.no_grad():
            for batch in pbar:
                data, target = batch[0], batch[-1]
                data, target = data.to(self.device), target.to(self.device)
                metadata = batch[1].to(self.device) if self.use_metadata else None
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                    output = self._forward(model, data, metadata)
                all_logits.append(output.float().cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        return np.concatenate(all_logits, axis=0), np.array(all_targets)

    # ------------------------------------------------------------------ checkpoints

    def _best_state(self, model, epoch, macro_f1, val_acc, val_loss, balanced_acc, weighted_f1):
        return {
            'experiment_id': self.experiment_id,
            'epoch': epoch,
            'best_macro_f1': macro_f1,
            'best_val_accuracy': val_acc,
            'best_val_loss': val_loss,
            'val_balanced_accuracy': balanced_acc,
            'val_weighted_f1': weighted_f1,
            'use_metadata': self.use_metadata,
            'metadata_preprocessor_state': self._preprocessor_state(),
            'model_state_dict': model.state_dict(),
        }

    # ------------------------------------------------------------------ temperature scaling

    @staticmethod
    def _apply_temperature(logits, temperature):
        """softmax(logits / T) with the max-subtraction stability trick. Returns
        probabilities with the same argmax as the unscaled softmax (T > 0 is a
        strictly monotone transform of the logits — verified at the call site)."""
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature!r}")
        shifted = (logits - logits.max(axis=1, keepdims=True)) / temperature
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit_temperature(self, model, val_loader, n_augments=4):
        """Post-hoc temperature scaling (Guo et al.): fit a single scalar T by
        minimizing CE(logits/T, labels) on the VALIDATION set with LBFGS. When
        use_tta is on, T is fit on the TTA-averaged logits — the exact pipeline
        that ships on the test set. Returns the fitted T (float > 0)."""
        if self.use_tta:
            _, val_targets, val_logits = self.predict_with_tta(
                model, val_loader, n_augments=n_augments)
        else:
            val_logits, val_targets = self.collect_logits(model, val_loader)

        logits_t = torch.tensor(val_logits, dtype=torch.float32, device=self.device)
        targets_t = torch.tensor(val_targets, dtype=torch.long, device=self.device)
        nll = nn.CrossEntropyLoss()

        # Optimize log(T), not T directly, so every LBFGS trial has T > 0.
        log_temperature = nn.Parameter(
            torch.log(torch.tensor(1.5, device=self.device)))
        optimizer = optim.LBFGS([log_temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nll(logits_t / log_temperature.exp(), targets_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(log_temperature.detach().exp().item())

    def fit_temperature_on_probs(self, val_probs, val_targets):
        """Temperature fit for probability ensembles: minimize CE(softmax(log p /
        T), labels) on the ensemble's averaged validation probabilities with
        LBFGS. Used to calibrate the F1-weighted ensemble, which averages member
        probabilities (not logits)."""
        log_probs_t = torch.tensor(
            np.log(np.clip(val_probs, 1e-12, 1.0)), dtype=torch.float32,
            device=self.device)
        targets_t = torch.tensor(val_targets, dtype=torch.long, device=self.device)
        nll = nn.CrossEntropyLoss()

        # As above, parameterize with log(T) to guarantee a positive scalar.
        log_temperature = nn.Parameter(
            torch.log(torch.tensor(1.5, device=self.device)))
        optimizer = optim.LBFGS([log_temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nll(log_probs_t / log_temperature.exp(), targets_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(log_temperature.detach().exp().item())

    @staticmethod
    def _check_temperature_invariance(unscaled_probs, scaled_probs):
        """Temperature scaling must change confidence but NEVER change argmax
        predictions (it is a monotone transform of the logits). Assert it; a
        violation means the scaling was applied to the wrong quantity."""
        same = np.array_equal(unscaled_probs.argmax(axis=1), scaled_probs.argmax(axis=1))
        if not same:
            raise AssertionError(
                "temperature scaling changed argmax predictions; "
                "the scaling pipeline is incompatible with its input representation"
            )
        return True

    # ------------------------------------------------------------------ checkpoints / ensemble

    def _trim_checkpoints(self):
        """Keep only the top-K best checkpoints (by val Macro-F1); delete worse
        ones from disk so the rotation never grows unbounded."""
        keep = max(1, self.keep_top_k_checkpoints)
        entries = sorted(self._best_checkpoints, key=lambda e: e['f1'], reverse=True)
        for entry in entries[keep:]:
            if os.path.exists(entry['path']):
                os.remove(entry['path'])
                print(f"   -> Removed {entry['path']} (outside top-{keep})")
        self._best_checkpoints = entries[:keep]

    def _rescan_checkpoints(self):
        """Rebuild the top-K list from the epoch-named best checkpoints found on
        disk (resume path: the in-memory list is not persisted in older
        checkpoints)."""
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
        self._best_checkpoints = entries
        self._trim_checkpoints()

    def _eligible_ensemble_members(self):
        """Once the blocks-4-5 (stage-2) unfreeze has fired, ensemble membership
        is restricted to checkpoints saved at/after that epoch — pre-stage-2
        checkpoints are never eligible for this ensemble."""
        if not self._stage2_unfrozen or self._stage2_epoch is None:
            return []
        return [
            entry for entry in self._best_checkpoints
            if entry['epoch'] >= self._stage2_epoch
        ]

    def _ensemble_probs(self, model, loader, members, criterion, use_tta=True,
                        n_augments=4):
        """F1-weighted averaging of member softmax probabilities on `loader`
        (weights = each member's saved val Macro-F1). Returns (avg_probs, targets,
        weight_sum). Used on BOTH val (for the usage decision) and test (for the
        final report) — the decision itself only ever uses the val numbers."""
        probs_sum = None
        targets = None
        weight_sum = 0.0
        loaded = 0
        for entry in members:
            path = entry['path']
            f1 = float(entry['f1'])
            if not os.path.exists(path):
                continue
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if use_tta:
                probs, t, _ = self.predict_with_tta(model, loader, n_augments=n_augments)
            else:
                _, _, _, t, probs = self.validate(model, loader, criterion,
                                                  epoch_label="ensemble")
            weight = max(f1, 1e-6)
            probs_sum = probs * weight if probs_sum is None else probs_sum + probs * weight
            weight_sum += weight
            targets = t
            loaded += 1
            print(f"    member {loaded}: {path} (val Macro-F1 {f1:.4f} @ epoch "
                  f"{checkpoint.get('epoch')}, weight {weight:.4f})")
        if probs_sum is None:
            return None, targets, 0.0
        return probs_sum / weight_sum, targets, weight_sum

    def _score_probs(self, probs, targets):
        """Macro-F1 / balanced accuracy / accuracy from probabilities."""
        preds = probs.argmax(axis=1)
        return {
            'macro_f1': f1_score(targets, preds, average='macro', zero_division=0),
            'balanced_acc': balanced_accuracy_score(targets, preds),
            'accuracy': 100.0 * float((preds == targets).mean()),
            'preds': preds,
        }

    # ------------------------------------------------------------------ plotting

    def plot_training_curves(self, history):
        """Training curves with the unfreeze boundaries marked (fixed blocks 6-8
        at epoch 6, plateau-gated blocks 4-5 at the epoch it actually fired)."""
        epochs = range(1, len(history['train_loss']) + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax1.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax1.axvline(x=6, linestyle='--', color='gray', label='Unfreeze blocks 6-8 (epoch 6)')
        if self._stage2_epoch is not None:
            ax1.axvline(x=self._stage2_epoch, linestyle='--', color='orange',
                        label=f'Unfreeze blocks 4-5 (epoch {self._stage2_epoch})')
        ax1.set_title('Training & Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax2.axvline(x=6, linestyle='--', color='gray')
        if self._stage2_epoch is not None:
            ax2.axvline(x=self._stage2_epoch, linestyle='--', color='orange')
        ax2.set_title('Training & Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        if 'lr' in history:
            ax3.plot(epochs, history['lr'], 'g-')
            ax3.set_title('Learning Rate (classifier group)')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True)

        ax4.plot(epochs, history['val_acc'], 'r-', label='Val accuracy')
        ax4.plot(epochs, history['val_macro_f1'], 'm-', label='Val Macro-F1')
        ax4.axvline(x=6, linestyle='--', color='gray')
        if self._stage2_epoch is not None:
            ax4.axvline(x=self._stage2_epoch, linestyle='--', color='orange')
        ax4.set_title('Validation Accuracy & Macro-F1')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Training curves saved as 'training_curves.png'")

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
            f'efficientnetb4_confusion_matrix_epoch_{epoch}.png',
        )

    def plot_final_confusion_matrix(self, y_true, y_pred):
        """Plot the final confusion matrix from the best model"""
        self._plot_confusion_matrix(
            y_true, y_pred,
            'Confusion Matrix - EfficientNetB4 (Final)',
            'confusion_matrix.png',
        )
        print("Confusion matrix saved as 'confusion_matrix.png'")

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
        print("ROC curves saved as 'roc_curves.png'")

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
        print("Precision-Recall curves saved as 'precision_recall_curves.png'")
        return pr_auc_per_class

    def plot_confidence_distribution(self, confidence, correct):
        """Plot confidence distributions for correct vs incorrect predictions"""
        plt.figure(figsize=(8, 6))
        plt.hist(confidence[correct], bins=20, alpha=0.6, label='Correct',
                 color='green', range=(0, 1))
        plt.hist(confidence[~correct], bins=20, alpha=0.6, label='Incorrect',
                 color='red', range=(0, 1))
        plt.xlabel('Confidence (max softmax probability)')
        plt.ylabel('Count')
        plt.title('Confidence Distribution - Correct vs Incorrect')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confidence distribution saved as 'confidence_distribution.png'")

    @staticmethod
    def _compute_ece(y_true, y_pred, probabilities, n_bins=10):
        """Expected Calibration Error: mean over bins of |accuracy - confidence|,
        weighted by bin size."""
        confidence = np.max(probabilities, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        n = len(confidence)
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
                ece += (size / n) * abs(accuracies[mask].mean() - confidence[mask].mean())
        return ece

    def plot_calibration_curve(self, y_true, y_pred, probabilities, n_bins=10):
        """Reliability diagram + ECE. Temperature scaling is applied to the
        shipped probabilities BEFORE this is called; the reported ECE therefore
        reflects the calibrated model."""
        confidence = np.max(probabilities, axis=1)
        accuracies = (y_pred == y_true).astype(float)

        bin_confidences = []
        bin_accuracies = []
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
            else:
                bin_conf = (lower + upper) / 2
                bin_acc = 0.0
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)

        ece = self._compute_ece(y_true, y_pred, probabilities, n_bins)

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
        print("Calibration curve saved as 'calibration_curve.png'")
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
        print("Generalization gap saved as 'generalization_gap.png'")

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
        print("Trainable parameters saved as 'trainable_parameters.png'")

    # ------------------------------------------------------------------ reporting

    def _final_metrics_report(self, targets, preds, probs, acc, best_epoch, ece,
                              model_label, ece_before=None, temperature=None,
                              argmax_invariant=True):
        """Final classification report, extended metrics, plots and summary printout."""
        print("\n" + "=" * 60)
        print("FINAL CLASSIFICATION REPORT")
        print("=" * 60)
        report = classification_report(
            targets, preds,
            target_names=self.class_names,
            digits=4,
            output_dict=True,
        )

        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        for class_name in self.class_names:
            metrics = report[class_name]
            print(f"{class_name:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {metrics['support']:<10}")

        print("-" * 80)
        print(f"{'Macro Avg':<10} {report['macro avg']['precision']:<12.4f} "
              f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
        print(f"{'Weighted Avg':<10} {report['weighted avg']['precision']:<12.4f} "
              f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}")

        balanced_acc = balanced_accuracy_score(targets, preds)
        macro_precision = precision_score(targets, preds, average='macro', zero_division=0)
        macro_recall = recall_score(targets, preds, average='macro', zero_division=0)
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)

        cm = confusion_matrix(targets, preds, labels=np.arange(self.num_classes))
        specificity_per_class = {}
        roc_auc_per_class = {}
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            specificity_per_class[class_name] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            binary_targets = (targets == i).astype(int)
            roc_auc_per_class[class_name] = roc_auc_score(binary_targets, probs[:, i])

        macro_roc_auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
        weighted_roc_auc = roc_auc_score(targets, probs, multi_class='ovr', average='weighted')

        self.plot_roc_curves(targets, probs)
        pr_auc_per_class = self.plot_pr_curves(targets, probs)
        macro_pr_auc = np.mean(list(pr_auc_per_class.values()))

        top2_preds = np.argsort(probs, axis=1)[:, -2:]
        top3_preds = np.argsort(probs, axis=1)[:, -3:]
        top2_acc = np.mean([t in p for t, p in zip(targets, top2_preds)])
        top3_acc = np.mean([t in p for t, p in zip(targets, top3_preds)])

        confidence = np.max(probs, axis=1)
        correct = preds == targets
        mean_confidence = np.mean(confidence)
        correct_confidence = np.mean(confidence[correct]) if np.any(correct) else 0.0
        incorrect_confidence = np.mean(confidence[~correct]) if np.any(~correct) else 0.0

        self.plot_confusion_matrix(targets, preds, best_epoch)
        self.plot_final_confusion_matrix(targets, preds)
        self.plot_confidence_distribution(confidence, correct)
        ece_plotted = self.plot_calibration_curve(targets, preds, probs)

        print("\n" + "=" * 60)
        print(f"FINAL MODEL EVALUATION ({model_label})")
        print("=" * 60)
        print(f"\nOverall Accuracy       : {acc / 100:.4f}")
        print(f"Balanced Accuracy      : {balanced_acc:.4f}")
        print(f"\nMacro Precision        : {macro_precision:.4f}")
        print(f"Macro Recall           : {macro_recall:.4f}")
        print(f"Macro F1               : {macro_f1:.4f}")
        print(f"\nWeighted F1            : {weighted_f1:.4f}")
        print(f"\nMacro ROC-AUC          : {macro_roc_auc:.4f}")
        print(f"Weighted ROC-AUC       : {weighted_roc_auc:.4f}")
        print(f"\nMacro PR-AUC           : {macro_pr_auc:.4f}")
        print(f"\nTop-1 Accuracy         : {acc / 100:.4f}")
        print(f"Top-2 Accuracy         : {top2_acc:.4f}")
        print(f"Top-3 Accuracy         : {top3_acc:.4f}")
        print(f"\nMean Confidence        : {mean_confidence:.4f}")
        print(f"Correct Confidence     : {correct_confidence:.4f}")
        print(f"Incorrect Confidence   : {incorrect_confidence:.4f}")
        if temperature is not None:
            print(f"\nTemperature scaling (Guo et al., fitted on validation):")
            print(f"    Fitted T           : {temperature:.4f}")
            if ece_before is not None:
                print(f"    ECE before scaling : {ece_before:.4f}")
            print(f"    ECE after scaling  : {ece:.4f}")
            print(f"    Argmax invariance  : "
                  f"{'PASS (predictions unchanged)' if argmax_invariant else 'FAIL'}")
        else:
            print(f"\nExpected Calibration Error : {ece:.4f}")

        print("\n------------------------------------------------------------")
        print("PER-CLASS METRICS")
        print("------------------------------------------------------------")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'Specificity':<12} "
              f"{'F1':<12} {'ROC-AUC':<10} {'PR-AUC':<10}")
        print("-" * 72)
        for class_name in self.class_names:
            m = report[class_name]
            print(f"{class_name:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                  f"{specificity_per_class[class_name]:<12.4f} {m['f1-score']:<12.4f} "
                  f"{roc_auc_per_class[class_name]:<10.4f} {pr_auc_per_class[class_name]:<10.4f}")

        return {
            'final_accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_roc_auc': macro_roc_auc,
            'weighted_roc_auc': weighted_roc_auc,
            'macro_pr_auc': macro_pr_auc,
            'top1_accuracy': acc,
            'top2_accuracy': top2_acc,
            'top3_accuracy': top3_acc,
            'mean_confidence': mean_confidence,
            'correct_confidence': correct_confidence,
            'incorrect_confidence': incorrect_confidence,
            'ece': ece_plotted,
            'ece_before_scaling': ece_before,
            'temperature': temperature,
            'temperature_argmax_invariant': argmax_invariant,
            'specificity_per_class': specificity_per_class,
            'roc_auc_per_class': roc_auc_per_class,
            'pr_auc_per_class': pr_auc_per_class,
            'classification_report': report,
        }

    # ------------------------------------------------------------------ startup

    def _print_startup(self, model, criterion, optimizer):
        """Full reproducibility printout (spec requirement 21)."""
        print("\n" + "=" * 72)
        print(f"EXPERIMENT: {self.experiment_id}")
        print("=" * 72)
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Grad accumulation: {self.grad_accum_steps} "
              f"(effective batch {self.batch_size * self.grad_accum_steps})")
        print(f"AMP: {'enabled (fp16 autocast + GradScaler)' if self.use_amp else 'disabled'}")
        print(f"Seed: {self.seed}")
        print(f"CuDNN: deterministic=True, benchmark=False (explicit reproducibility "
              f"choice; slightly slower than benchmark mode)")
        print(f"Optimizer: AdamW (weight_decay={self.weight_decay})")
        print("Learning rates per parameter group (peak LRs, restored recipe):")
        for name, base in [('classifier', self.lr_head),
                           ('blocks 6-8', self.lr_blocks_6_8),
                           ('blocks 4-5', self.lr_blocks_4_5)]:
            print(f"    {name:<12} lr={base:.2e}")
        print("Scheduler: ReduceLROnPlateau(mode='max', factor=0.5, patience=5, "
              "min_lr=1e-7) on validation Macro-F1, stepped after each validation "
              "phase (warmup+cosine experiment closed)")
        print(f"Loss: {type(criterion).__name__}"
              + (f" (gamma={self.focal_gamma}, alpha=1/sqrt(count) mean-normalized)"
                 if self.loss_mode == 'focal' else
                 " (weight=1/sqrt(count) mean-normalized)" if self.loss_mode == 'weighted_ce'
                 else ""))
        if self.class_weights is not None:
            print("Class weights / focal alpha:")
            for name, w in zip(self.class_names, self.class_weights.cpu().numpy()):
                print(f"    {name}: {w:.4f}")
        print(f"Augmentation (train): aspect-preserving Resize({int(self.image_size * 256 / 224)}) "
              f"-> RandomResizedCrop({self.image_size}, scale=(0.8, 1.0), ratio=(0.9, 1.1)) "
              f"-> HFlip(0.5) -> VFlip(0.5) -> Rotation(15) -> "
              f"ColorJitter(b=0.15, c=0.15, s=0.15, h=0.02) -> ToTensor -> "
              f"Normalize(ImageNet mean={IMAGENET_MEAN}, std={IMAGENET_STD})"
              + (" -> RandomErasing(p=0.1)" if self.use_random_erasing else "")
              + (" -> MixUp/CutMix (soft-target focal, batch-random pick)"
                 if (self.use_mixup or self.use_cutmix) else ""))
        print(f"Augmentation (val/test): aspect-preserving Resize({self.eval_resize}) "
              f"-> CenterCrop({self.image_size}) -> ToTensor -> "
              f"Normalize(ImageNet mean={IMAGENET_MEAN}, std={IMAGENET_STD})")
        print("Unfreezing schedule (restored recipe):")
        print("    Epochs 1-5: head")
        print("    Epoch 6+:   head + blocks 6-8 (fixed epoch, consistently worked)")
        print("    Blocks 4-5: plateau-gated — fires ONCE after "
              f"{self.stage2_unfreeze_patience} epochs without a val Macro-F1 "
              "improvement, with BN running stats frozen")
        print("    Blocks 0-3 remain frozen")
        print(f"Imbalance: plain shuffle + {self.loss_mode}")
        print(f"Model selection: validation Macro-F1 (also tracking validation accuracy)")
        print(f"Temperature scaling: Guo et al., T fitted on validation "
              f"({'TTA-averaged ' if self.use_tta else ''}logits)")
        print(f"Ensemble: top-{self.keep_top_k_checkpoints} "
              f"{'on' if self.use_ensemble else 'off'} (F1-weighted, post-stage-2 "
              "members only, val-gated usage)")
        print(f"TTA: {'on (final evaluation only)' if self.use_tta else 'off'}"
              + (f", multi-scale (second scale {self.tta_second_scale}px)"
                 if self.use_multiscale_tta else ""))
        if self.use_metadata:
            p = self.preprocessor
            print(f"Metadata branch: ENABLED (fused via MLP; classifier input "
                  f"{self.metadata_dim + 32} metadata-aware)")
            print(f"    Age median: {p.age_median:.4f} | mean: {p.age_mean:.4f} | "
                  f"std: {p.age_std:.4f} (fitted on TRAIN split only)")
            print(f"    Sex categories: {p.sex_categories}")
            print(f"    Localization categories ({len(p.localization_categories)}): "
                  f"{p.localization_categories}")
            print(f"    Final metadata dimension: {p.dim}")
            print("    Leakage: preprocessing fitted on train only; val/test transform "
                  "never alters fitted state (PASS at startup)")
        else:
            print("Metadata branch: disabled — image-only baseline "
                  "(no metadata dependency in the model path)")
        print("=" * 72)

    # ------------------------------------------------------------------ main loop

    def train(self):
        """Full training pipeline: gradual unfreezing (fixed blocks 6-8 @ epoch 6,
        plateau-gated blocks 4-5), ReduceLROnPlateau on val Macro-F1, top-K
        checkpoint rotation, Macro-F1 + accuracy model selection, resume support,
        temperature scaling + val-gated F1-weighted ensemble for the final test
        evaluation."""
        set_seed(self.seed)

        if self.resume_checkpoint is None:
            # Fresh run: clear stale epoch-named best checkpoints from any previous
            # experiment so the top-K rotation/ensemble only covers this run.
            for stale in glob.glob('efficientnetb4_best_model_epoch*.pth'):
                os.remove(stale)
                print(f"   -> Removed stale best checkpoint from a previous run: {stale}")
            self._best_checkpoints = []

        train_loader, val_loader, test_loader = self.create_dataloaders()
        model = self.build_model()
        criterion = self._build_criterion()

        # Explicit param groups up front; frozen groups are never updated because
        # their params have requires_grad=False (no gradient -> AdamW skips them).
        optimizer = optim.AdamW(self._build_param_groups(model))
        scheduler = self._build_scheduler(optimizer)

        self._print_startup(model, criterion, optimizer)

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_macro_f1': [], 'val_balanced_acc': [], 'val_weighted_f1': [],
            'lr': [],
            'trainable_params': [],
        }

        best_macro_f1 = -1.0
        best_epoch = None
        best_val_acc = 0.0
        best_acc_epoch = None
        patience_counter = 0
        start_epoch = 0

        if self.resume_checkpoint is not None:
            if not os.path.exists(self.resume_checkpoint):
                raise FileNotFoundError(
                    f"Cannot resume training: checkpoint file not found: {self.resume_checkpoint}"
                )
            checkpoint = torch.load(self.resume_checkpoint, map_location='cpu',
                                    weights_only=False)
            self._check_resume_compatible(checkpoint)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            best_macro_f1 = checkpoint['best_metric']
            best_epoch = checkpoint['best_epoch']
            best_val_acc = checkpoint['best_val_accuracy']
            best_acc_epoch = checkpoint['best_acc_epoch']
            patience_counter = checkpoint['patience_counter']
            history = checkpoint['history']
            self._stage2_unfrozen = checkpoint.get('stage2_unfrozen', False)
            self._stage2_epoch = checkpoint.get('stage2_epoch')
            self._best_checkpoints = checkpoint.get('best_checkpoints', [])

            torch.set_rng_state(checkpoint['rng_state'])
            if checkpoint.get('cuda_rng_state') is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
                except TypeError:
                    torch.cuda.set_rng_state_all(
                        [s.cuda() for s in checkpoint['cuda_rng_state']])
            if checkpoint.get('numpy_rng_state') is not None:
                np.random.set_state(checkpoint['numpy_rng_state'])
            if checkpoint.get('python_rng_state') is not None:
                random.setstate(checkpoint['python_rng_state'])

            # Metadata mode: the checkpoint's train-fitted preprocessing state must
            # match the fresh train-only fit (deterministic -> must be identical).
            if self.use_metadata:
                ckpt_state = checkpoint.get('metadata_preprocessor_state')
                if ckpt_state is None:
                    raise ValueError(
                        "Resume checkpoint from a metadata run is missing "
                        "metadata_preprocessor_state — refusing to resume."
                    )
                if json.dumps(ckpt_state, sort_keys=True, default=float) \
                        != self._fresh_preprocessor_state:
                    raise ValueError(
                        "Metadata preprocessing state in the resume checkpoint differs "
                        "from a fresh train-only fit. The training data must have "
                        "changed — refusing to resume to avoid inconsistent features."
                    )
                print("Metadata preprocessing state restored from checkpoint "
                      "(identical to fresh train-only fit).")

            start_epoch = checkpoint['epoch']  # next epoch to run (0-indexed)
            print("\n" + "=" * 60)
            print("RESUMING TRAINING")
            print(f"Checkpoint: {self.resume_checkpoint}")
            print(f"Resuming from epoch: {start_epoch + 1}")
            print(f"Previous best validation Macro-F1: {best_macro_f1:.4f} @ epoch {best_epoch}")
            print(f"Previous best validation accuracy: {best_val_acc:.2f}% @ epoch {best_acc_epoch}")
            print(f"Stage-2 (blocks 4-5) unfrozen: {self._stage2_unfrozen}"
                  + (f" @ epoch {self._stage2_epoch}" if self._stage2_epoch else ""))
            print("=" * 60)
        else:
            print("\nNo checkpoint specified → starting from epoch 1")

        # Rebuild the top-K list from disk: the in-memory list may predate the
        # resume or files may have been trimmed externally.
        if self._best_checkpoints:
            self._rescan_checkpoints()
        # Re-apply stage-2 BN running-stat freeze on resume (modules are new).
        if self._stage2_unfrozen:
            self._freeze_bn_stats(model, [4, 5])

        # Stage-aware trainable flags (re-applied on resume so requires_grad
        # always matches the current stage).
        self._set_trainable(model, start_epoch)
        if start_epoch >= self.num_epochs:
            print(f"Checkpoint already reached epoch {self.num_epochs}; no epochs left to train.")

        print("\n" + "=" * 60)
        print("Training with Gradual Unfreezing (restored recipe)")
        print("=" * 60)
        print("    Epochs 1-5: head")
        print("    Epoch 6+:   head + blocks 6-8")
        print(f"    Blocks 4-5: plateau-gated (after {self.stage2_unfreeze_patience} epochs "
              f"without val Macro-F1 improvement; BN running stats frozen)")
        print("    Blocks 0-3 remain frozen")
        print("=" * 60)

        start_time = time.time()
        try:
            for epoch in range(start_epoch, self.num_epochs):
                # ===== UNFREEZE TRANSITIONS =====
                # Blocks 6-8: fixed at 1-based epoch 6 (0-based idx 5) — this fixed
                # epoch consistently worked, so it stays fixed.
                if epoch == 5 and not self._stage2_unfrozen:
                    print(f"\nUNFREEZING blocks 6-8 at epoch {epoch + 1} "
                          f"(head + blocks 6-8 from now on)...")
                    self._set_trainable(model, epoch)
                # Blocks 4-5: plateau-gated, fires ONCE — only after blocks 6-8 have
                # had room to help and val Macro-F1 has gone stage2_unfreeze_patience
                # epochs without improving (reuses the early-stopping patience
                # counter; no duplicate tracking). BN running stats in blocks 4-5
                # are pinned to eval mode on firing (small-batch stabilizer).
                if (not self._stage2_unfrozen and epoch > 5
                        and patience_counter >= self.stage2_unfreeze_patience):
                    print(f"\nUNFREEZING blocks 4-5 at epoch {epoch + 1} "
                          f"(no val Macro-F1 improvement for {patience_counter} "
                          f"epochs)...")
                    self._stage2_unfrozen = True
                    self._stage2_epoch = epoch + 1
                    self._set_trainable(model, epoch)
                    self._freeze_bn_stats(model, [4, 5])

                train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                val_loss, val_acc, val_preds, val_targets, _ = self.validate(
                    model, val_loader, criterion, epoch_label=f"{epoch + 1}/{self.num_epochs}")

                val_macro_f1 = f1_score(val_targets, val_preds, average='macro',
                                        zero_division=0)
                val_balanced_acc = balanced_accuracy_score(val_targets, val_preds)
                val_weighted_f1 = f1_score(val_targets, val_preds, average='weighted',
                                           zero_division=0)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_macro_f1'].append(val_macro_f1)
                history['val_balanced_acc'].append(val_balanced_acc)
                history['val_weighted_f1'].append(val_weighted_f1)
                history['lr'].append(optimizer.param_groups[0]['lr'])
                history['trainable_params'].append(trainable_params)

                # ===== EPOCH SUMMARY =====
                print("\n" + "=" * 80)
                print(f"EPOCH {epoch + 1}/{self.num_epochs} SUMMARY")
                print("=" * 80)
                print(f"Training   -> Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
                print(f"Validation -> Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}% "
                      f"| Macro-F1: {val_macro_f1:.4f} | Balanced Acc: {val_balanced_acc:.4f} "
                      f"| Weighted F1: {val_weighted_f1:.4f}")
                for group in optimizer.param_groups:
                    print(f"    {group['name']:<12} lr={group['lr']:.2e}")

                # ===== SCHEDULER STEP =====
                # ReduceLROnPlateau tracks the same metric used for model
                # selection: val Macro-F1 (mode='max'). Must stay before the
                # last-checkpoint save so the saved scheduler state points at
                # the next epoch on resume.
                lr_before = [g['lr'] for g in optimizer.param_groups]
                scheduler.step(val_macro_f1)
                lr_after = [g['lr'] for g in optimizer.param_groups]
                if lr_after != lr_before:
                    print("    ReduceLROnPlateau: LR reduced (factor 0.5)")

                # ===== BEST-MODEL UPDATE (primary metric: validation Macro-F1) =====
                if val_macro_f1 > best_macro_f1:
                    improvement = val_macro_f1 - best_macro_f1
                    best_macro_f1 = val_macro_f1
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state = self._best_state(
                        model, best_epoch, best_macro_f1, val_acc, val_loss,
                        val_balanced_acc, val_weighted_f1)
                    # Rotating top-K filenames: keep the best K by Macro-F1 on
                    # disk for the ensemble, delete worse ones. The canonical
                    # unqualified filename is kept as a copy of the single best
                    # so resume/external tooling is unaffected.
                    rotating_path = f'efficientnetb4_best_model_epoch{best_epoch}.pth'
                    torch.save(best_state, rotating_path)
                    self._best_checkpoints = [
                        e for e in self._best_checkpoints if e['path'] != rotating_path]
                    self._best_checkpoints.append(
                        {'path': rotating_path, 'f1': best_macro_f1, 'epoch': best_epoch})
                    self._trim_checkpoints()
                    torch.save(best_state, 'efficientnetb4_best_model.pth')
                    print(f"SAVING MODEL -> Epoch {epoch + 1} | New Best Macro F1: "
                          f"{best_macro_f1:.4f} (+{improvement:.4f}) | Acc: {val_acc:.2f}%")
                    self.plot_confusion_matrix(val_targets, val_preds, epoch + 1)
                else:
                    patience_counter += 1
                    print(f"NO IMPROVEMENT -> Epoch {epoch + 1} | "
                          f"Patience: {patience_counter}/{self.early_stop_patience}")

                # ===== BEST-BY-ACCURACY CHECKPOINT (secondary, informative) =====
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_acc_epoch = epoch + 1
                    torch.save(self._best_state(
                        model, best_acc_epoch, best_macro_f1, val_acc, val_loss,
                        val_balanced_acc, val_weighted_f1),
                        'efficientnetb4_best_accuracy_model.pth')
                    print(f"SAVING ACC-BEST MODEL -> Epoch {epoch + 1} | "
                          f"New Best Val Accuracy: {val_acc:.2f}%")

                # ===== LAST CHECKPOINT (resume support; overwritten every epoch) =====
                torch.save({
                    'experiment_id': self.experiment_id,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'best_metric': best_macro_f1,
                    'best_epoch': best_epoch,
                    'best_val_accuracy': best_val_acc,
                    'best_acc_epoch': best_acc_epoch,
                    'patience_counter': patience_counter,
                    'history': history,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None,
                    'numpy_rng_state': np.random.get_state(),
                    'python_rng_state': random.getstate(),
                    'loss_mode': self.loss_mode,
                    'image_size': self.image_size,
                    'batch_size': self.batch_size,
                    'grad_accum_steps': self.grad_accum_steps,
                    'num_epochs': self.num_epochs,
                    'lr_head': self.lr_head,
                    'lr_blocks_6_8': self.lr_blocks_6_8,
                    'lr_blocks_4_5': self.lr_blocks_4_5,
                    'seed': self.seed,
                    'use_metadata': self.use_metadata,
                    'metadata_preprocessor_state': self._preprocessor_state(),
                    'stage2_unfrozen': self._stage2_unfrozen,
                    'stage2_epoch': self._stage2_epoch,
                    'stage2_unfreeze_patience': self.stage2_unfreeze_patience,
                    'use_ensemble': self.use_ensemble,
                    'keep_top_k_checkpoints': self.keep_top_k_checkpoints,
                    'best_checkpoints': self._best_checkpoints,
                }, 'efficientnetb4_last_checkpoint.pth')

                # ===== EARLY STOPPING (validation Macro-F1) =====
                if patience_counter >= self.early_stop_patience:
                    print(f"\nEARLY STOPPING triggered at epoch {epoch + 1}")
                    break
        except torch.cuda.OutOfMemoryError:
            print("\nOUT OF MEMORY during training. Fall back to "
                  "batch_size=8 + grad_accum_steps=2 (effective batch stays 16) "
                  "and rerun. The last checkpoint covers completed epochs only — "
                  "delete it to start fresh.")
            raise

        total_training_time = time.time() - start_time

        # ===== FINAL RESULTS =====
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total time: {total_training_time:.2f} seconds "
              f"(~{total_training_time / max(1, len(history['train_loss'])):.1f} s/epoch)")
        print(f"Best Validation Macro F1: {best_macro_f1:.4f} (epoch {best_epoch})")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}% (epoch {best_acc_epoch})")
        if self._stage2_unfrozen:
            print(f"Stage-2 (blocks 4-5) unfreeze fired at epoch {self._stage2_epoch}")

        # Evaluate the best-Macro-F1 model on the held-out test set (touched at the end).
        best_model_path = 'efficientnetb4_best_model.pth'
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"{best_model_path} not found — cannot evaluate.")
        best_checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        best_epoch = best_checkpoint['epoch']

        # Single-pass no-TTA baseline first so the TTA delta is visible.
        _, no_tta_acc, no_tta_preds, no_tta_targets, no_tta_probs = self.validate(
            model, test_loader, criterion, epoch_label="test")
        no_tta_f1 = f1_score(no_tta_preds, no_tta_targets, average='macro', zero_division=0)

        # ===== SINGLE BEST MODEL: val (temperature + gate) / test (report) =====
        if self.use_tta:
            val_probs_single, val_targets_single, _ = self.predict_with_tta(
                model, val_loader)
            test_probs_single, test_targets, test_logits_single = self.predict_with_tta(
                model, test_loader)
        else:
            _, _, _, val_targets_single, val_probs_single = self.validate(
                model, val_loader, criterion, epoch_label="val")
            test_probs_single, test_targets = no_tta_probs, no_tta_targets
            _, test_logits_single = self.collect_logits(model, test_loader,
                                                        epoch_label="test")

        single_val_f1 = self._score_probs(val_probs_single, val_targets_single)['macro_f1']

        # ===== MULTI-SCALE TTA INCREMENTAL GAIN (Phase 5, opt-in) =====
        tta_single_scale_f1 = None
        if self.use_tta and self.use_multiscale_tta:
            probs_ss, _, _ = self.predict_with_tta(
                model, test_loader, scales=[self.image_size])
            tta_single_scale_f1 = f1_score(
                probs_ss.argmax(axis=1), test_targets, average='macro',
                zero_division=0)
            f1_ms = f1_score(test_probs_single.argmax(axis=1), test_targets,
                             average='macro', zero_division=0)
            print(f"\nTTA scale gain: single-scale ({self.image_size}px) "
                  f"Macro-F1 {tta_single_scale_f1:.4f} -> multi-scale "
                  f"(+{self.tta_second_scale}px) Macro-F1 {f1_ms:.4f} "
                  f"(gain {f1_ms - tta_single_scale_f1:+.4f})")

        # ===== TEMPERATURE SCALING (Guo et al.) — fit T on VALIDATION =====
        print("\n" + "=" * 60)
        print("TEMPERATURE SCALING (fitted on validation, never on test)")
        print("=" * 60)
        temperature = self.fit_temperature(model, val_loader)
        print(f"    Fitted T: {temperature:.4f} "
              f"(CE(logits/T, y) via LBFGS on val"
              + (", TTA-averaged logits" if self.use_tta else ", single-pass logits")
              + ")")

        scaled_single = self._apply_temperature(test_logits_single, temperature)
        ece_before_single = self._compute_ece(
            test_targets, test_probs_single.argmax(axis=1), test_probs_single)
        ece_after_single = self._compute_ece(
            test_targets, scaled_single.argmax(axis=1), scaled_single)
        invariant_single = self._check_temperature_invariance(test_probs_single,
                                                              scaled_single)
        print(f"    ECE before scaling: {ece_before_single:.4f}")
        print(f"    ECE after scaling : {ece_after_single:.4f}")
        print(f"    Argmax invariance : "
              f"{'PASS (no prediction changed)' if invariant_single else 'FAIL'}")

        # ===== ENSEMBLE (top-K, F1-weighted, post-stage-2, val-gated) =====
        ensemble_result = None
        use_ensemble_final = False
        if self.use_ensemble:
            members = self._eligible_ensemble_members()
            if not members:
                print("\nENSEMBLE: no top-K best checkpoints available — using "
                      "single best model.")
            else:
                print("\n" + "=" * 60)
                print("ENSEMBLE EVALUATION (top-K best checkpoints, F1-weighted)")
                print("=" * 60)
                print("Eligible members (post-stage-2 only once stage 2 fired):")
                val_probs_ens, val_targets_ens, _ = self._ensemble_probs(
                    model, val_loader, members, criterion)
                test_probs_ens, test_targets_ens, _ = self._ensemble_probs(
                    model, test_loader, members, criterion)
                if val_probs_ens is None:
                    print("    No ensemble member could be loaded — using single best.")
                else:
                    ens_val = self._score_probs(val_probs_ens, val_targets_ens)
                    ens_test = self._score_probs(test_probs_ens, test_targets_ens)
                    # Fit the ensemble's own temperature on its VALIDATION probs.
                    temp_ens = self.fit_temperature_on_probs(
                        val_probs_ens, val_targets_ens)
                    scaled_ens = self._apply_temperature(
                        np.log(np.clip(test_probs_ens, 1e-12, 1.0)), temp_ens)
                    ece_before_ens = self._compute_ece(
                        test_targets_ens, test_probs_ens.argmax(axis=1), test_probs_ens)
                    ece_after_ens = self._compute_ece(
                        test_targets_ens, scaled_ens.argmax(axis=1), scaled_ens)
                    invariant_ens = self._check_temperature_invariance(
                        test_probs_ens, scaled_ens)

                    # ===== VAL-ONLY USAGE DECISION (never looks at test) =====
                    print("\n    VALIDATION GATE (decision on validation only):")
                    print(f"        Single best val Macro-F1: {single_val_f1:.4f}")
                    print(f"        Ensemble val Macro-F1   : {ens_val['macro_f1']:.4f}")
                    if ens_val['macro_f1'] > single_val_f1:
                        use_ensemble_final = True
                        print("        Decision: ENSEMBLE used on test "
                              "(wins on validation)")
                    else:
                        print("        Decision: SINGLE BEST used on test "
                              "(ensemble does not beat it on validation)")
                    if use_ensemble_final:
                        print("    Ensemble temperature (fitted on val probs): "
                              f"{temp_ens:.4f}")
                        print(f"    Ensemble ECE before scaling: {ece_before_ens:.4f}")
                        print(f"    Ensemble ECE after scaling : {ece_after_ens:.4f}")
                    ensemble_result = {
                        'used': use_ensemble_final,
                        'val_macro_f1': ens_val['macro_f1'],
                        'test_macro_f1': ens_test['macro_f1'],
                        'test_balanced_acc': ens_test['balanced_acc'],
                        'members': [e['path'] for e in members],
                        'temperature': temp_ens,
                        'ece_before': ece_before_ens,
                        'ece_after': ece_after_ens,
                        'argmax_invariant': invariant_ens,
                    }
                print("=" * 60)

        # ===== FINAL REPORTED PREDICTIONS (temperature-scaled, ensemble or single) =====
        if use_ensemble_final and ensemble_result is not None:
            final_probs = scaled_ens
            final_targets = test_targets_ens
            final_preds = final_probs.argmax(axis=1)
            final_acc = 100.0 * float((final_preds == final_targets).mean())
            tta_f1 = f1_score(final_preds, final_targets, average='macro',
                              zero_division=0)
            ece = ece_after_ens
            ece_before = ece_before_ens
            temperature_final = ensemble_result['temperature']
            argmax_invariant = ensemble_result['argmax_invariant']
            model_label = "F1-weighted ensemble (TTA)" if self.use_tta \
                else "F1-weighted ensemble (no TTA)"
        else:
            final_probs = scaled_single
            final_targets = test_targets
            final_preds = final_probs.argmax(axis=1)
            final_acc = 100.0 * float((final_preds == final_targets).mean())
            tta_f1 = f1_score(final_preds, final_targets, average='macro',
                              zero_division=0)
            ece = ece_after_single
            ece_before = ece_before_single
            temperature_final = temperature
            argmax_invariant = invariant_single
            model_label = "single best (TTA)" if self.use_tta \
                else "single best (no TTA)"

        if self.use_tta:
            print(f"\nTest Macro-F1: no-TTA {no_tta_f1:.4f} -> TTA {tta_f1:.4f}")

        self.plot_training_curves(history)
        self.plot_trainable_parameters(history)
        self.plot_generalization_gap(history)

        report_metrics = self._final_metrics_report(
            final_targets, final_preds, final_probs, final_acc, best_epoch, ece,
            model_label, ece_before=ece_before, temperature=temperature_final,
            argmax_invariant=argmax_invariant)

        # ===== EXPERIMENT CONFIGURATION RECAP (as run) =====
        print("\n" + "=" * 60)
        print(f"EXPERIMENT CONFIGURATION (as run) — {self.experiment_id}")
        print("=" * 60)
        print(f"Model: EfficientNet-B4 (ImageNet pretrained, correct ImageNet normalization)")
        print(f"Dataset: HAM10000 (lesion-level split, unchanged) | "
              f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | "
              f"Test: {len(test_loader.dataset)}")
        print(f"Image size: {self.image_size}x{self.image_size} | "
              f"Eval resize: {self.eval_resize} (shorter side, aspect preserved)")
        print(f"Batch size: {self.batch_size} | Grad accum: {self.grad_accum_steps} "
              f"(effective batch {self.batch_size * self.grad_accum_steps}) | "
              f"Max epochs: {self.num_epochs}")
        print(f"Optimizer: AdamW (weight_decay={self.weight_decay}) | "
              f"LRs: head {self.lr_head:.1e}, blocks 6-8 {self.lr_blocks_6_8:.1e}, "
              f"blocks 4-5 {self.lr_blocks_4_5:.1e}")
        print(f"Scheduler: ReduceLROnPlateau(mode='max', factor=0.5, patience=5, "
              f"min_lr=1e-7) on val Macro-F1 | Loss: {self.loss_mode} | AMP: {self.use_amp}")
        print(f"Unfreezing: head (epochs 1-5) -> +blocks 6-8 (epoch 6+) -> +blocks 4-5 "
              f"(plateau-gated: {self.stage2_unfreeze_patience} epochs without val "
              f"Macro-F1 improvement, BN running stats frozen"
              + (f", fired @ epoch {self._stage2_epoch}" if self._stage2_epoch else "")
              + "); blocks 0-3 frozen")
        print(f"Model selection: validation Macro-F1 (epoch {best_epoch}) | "
              f"validation accuracy (epoch {best_acc_epoch})")
        print(f"TTA: {'enabled (final evaluation only)' if self.use_tta else 'disabled'}"
              + (f", multi-scale (second scale {self.tta_second_scale}px)"
                 if self.use_multiscale_tta else "")
              + f" | Ensemble: {'on' if self.use_ensemble else 'off'} "
              f"(top-{self.keep_top_k_checkpoints}, F1-weighted, "
              f"{'USED on test' if use_ensemble_final else 'NOT used on test (lost val gate)'})"
              + f" | Temperature scaling: on (T={temperature_final:.4f})")
        print(f"MixUp: {'on' if self.use_mixup else 'off'} | "
              f"CutMix: {'on' if self.use_cutmix else 'off'}")
        print(f"Random seed: {self.seed} | Class order: {self.class_names}")
        if self.use_metadata:
            print(f"Metadata: ENABLED ({METADATA_EXPERIMENT_ID}) — dim {self.metadata_dim}, "
                  f"MLP -> 64 -> 32, fusion concat(image features, metadata embedding)")
        else:
            print(f"Metadata: disabled (image-only baseline, {EXPERIMENT_ID})")
        print(f"Best validation Macro F1: {best_macro_f1:.4f} @ epoch {best_epoch}")
        print(f"Test Macro-F1: no-TTA {no_tta_f1:.4f} -> final {tta_f1:.4f} | "
              f"Test accuracy: {final_acc:.2f}%")
        if ensemble_result is not None:
            print(f"Ensemble: val F1 {ensemble_result['val_macro_f1']:.4f} | "
                  f"test F1 {ensemble_result['test_macro_f1']:.4f} | "
                  f"members {len(ensemble_result['members'])}")
        print("=" * 60)

        # ===== SAVE FULL METRICS =====
        metrics = {
            'experiment_id': self.experiment_id,
            'loss_mode': self.loss_mode,
            'focal_gamma': self.focal_gamma,
            'image_size': self.image_size,
            'eval_resize': self.eval_resize,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'grad_accum_steps': self.grad_accum_steps,
            'weight_decay': self.weight_decay,
            'lr_head': self.lr_head,
            'lr_blocks_6_8': self.lr_blocks_6_8,
            'lr_blocks_4_5': self.lr_blocks_4_5,
            'use_amp': self.use_amp,
            'use_tta': self.use_tta,
            'use_multiscale_tta': self.use_multiscale_tta,
            'tta_second_scale': self.tta_second_scale,
            'use_random_erasing': self.use_random_erasing,
            'use_mixup': self.use_mixup,
            'use_cutmix': self.use_cutmix,
            'use_ensemble': self.use_ensemble,
            'keep_top_k_checkpoints': self.keep_top_k_checkpoints,
            'use_metadata': self.use_metadata,
            'metadata_dim': self.metadata_dim,
            'metadata_feature_names': self.preprocessor.feature_names
            if self.use_metadata else None,
            'metadata_preprocessor_state': self._preprocessor_state(),
            'num_epochs': self.num_epochs,
            'early_stop_patience': self.early_stop_patience,
            'class_names': self.class_names,
            'class_counts': dict(zip(self.class_names, self.class_counts)),
            'split_sizes': {
                'train': len(train_loader.dataset),
                'val': len(val_loader.dataset),
                'test': len(test_loader.dataset),
            },
            'history': history,
            'best_macro_f1': best_macro_f1,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_acc,
            'best_acc_epoch': best_acc_epoch,
            'no_tta_macro_f1': no_tta_f1,
            'tta_macro_f1': tta_f1,
            'tta_single_scale_macro_f1': tta_single_scale_f1,
            'final_accuracy': final_acc,
            'total_training_time': total_training_time,
            'epochs_trained': len(history['train_loss']),
            'stage2_unfrozen': self._stage2_unfrozen,
            'stage2_epoch': self._stage2_epoch,
            'stage2_unfreeze_patience': self.stage2_unfreeze_patience,
            'ensemble_used': use_ensemble_final,
            'ensemble_val_macro_f1': ensemble_result['val_macro_f1']
            if ensemble_result is not None else None,
            'ensemble_test_macro_f1': ensemble_result['test_macro_f1']
            if ensemble_result is not None else None,
            'ensemble_test_balanced_acc': ensemble_result['test_balanced_acc']
            if ensemble_result is not None else None,
            'ensemble_members': ensemble_result['members'] if ensemble_result else None,
            'temperature': temperature_final,
        }
        metrics.update(report_metrics)
        torch.save(metrics, 'efficientnetb4_training_metrics.pth')
        print(f"Metrics saved to 'efficientnetb4_training_metrics.pth' "
              f"({self.experiment_id})")

        return model, history, metrics

    def _check_resume_compatible(self, checkpoint):
        """Refuse to resume a checkpoint from a different experiment/config so a
        stale file can never silently corrupt a fresh run's results. num_epochs is
        the one exception: resuming with MORE epochs is allowed (ReduceLROnPlateau
        has no fixed schedule tail to extend); shrinking is refused."""
        expected = {
            'experiment_id': self.experiment_id,
            'loss_mode': self.loss_mode,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'grad_accum_steps': self.grad_accum_steps,
            'lr_head': self.lr_head,
            'lr_blocks_6_8': self.lr_blocks_6_8,
            'lr_blocks_4_5': self.lr_blocks_4_5,
            'seed': self.seed,
        }
        # Old image-only checkpoints predate use_metadata -> treat missing key as False.
        # Checked FIRST so cross-mode resume failures get the most actionable message.
        if checkpoint.get('use_metadata', False) != self.use_metadata:
            raise ValueError(
                f"Cannot resume: checkpoint use_metadata="
                f"{checkpoint.get('use_metadata', False)} but this run uses "
                f"use_metadata={self.use_metadata}. Image-only and image+metadata "
                f"checkpoints are not interchangeable. Delete {self.resume_checkpoint} "
                f"to start fresh."
            )
        mismatches = {k: (checkpoint.get(k), v) for k, v in expected.items()
                      if checkpoint.get(k) != v}
        if mismatches:
            detail = ', '.join(f'{k}: ckpt={a} run={b}' for k, (a, b) in mismatches.items())
            raise ValueError(
                f"Cannot resume: checkpoint config differs from this run ({detail}). "
                f"This checkpoint belongs to a different experiment or configuration. "
                f"Delete {self.resume_checkpoint} to start fresh."
            )
        if checkpoint.get('num_epochs', 0) > self.num_epochs:
            raise ValueError(
                f"Cannot resume: checkpoint was created with num_epochs="
                f"{checkpoint['num_epochs']} but this run has num_epochs="
                f"{self.num_epochs}. Extending past a finished schedule is refused; "
                f"use num_epochs >= {checkpoint['num_epochs']} to continue."
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EfficientNet-B4 HAM10000 trainer (image-only or image+metadata).")
    parser.add_argument(
        "--use-metadata", action="store_true",
        help=f"enable image+metadata fusion ({METADATA_EXPERIMENT_ID}); default is "
             f"the image-only baseline ({EXPERIMENT_ID})")
    parser.add_argument(
        "--metadata-csv", default=DEFAULT_METADATA_CSV,
        help="ground-truth CSV with metadata columns (default: %(default)s)")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="maximum epochs (default: %(default)s)")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="peak head LR (default: %(default)s); blocks 6-8 = "
                             "head*0.1, blocks 4-5 = head*0.05 unless overridden")
    parser.add_argument("--stage2-unfreeze-patience", type=int, default=4,
                        help="epochs without val Macro-F1 improvement before the "
                             "blocks-4-5 unfreeze fires (default: %(default)s)")
    parser.add_argument("--use-ensemble", dest="use_ensemble", action="store_true",
                        default=True, help="F1-weighted top-K ensemble (default: on)")
    parser.add_argument("--no-ensemble", dest="use_ensemble", action="store_false",
                        help="disable the checkpoint ensemble")
    parser.add_argument("--keep-top-k", type=int, default=3,
                        help="top-K best checkpoints kept for the ensemble "
                             "(default: %(default)s)")
    parser.add_argument("--use-mixup", action="store_true",
                        help="opt-in MixUp (alpha=0.2, soft-target focal)")
    parser.add_argument("--use-cutmix", action="store_true",
                        help="opt-in CutMix (alpha=1.0, soft-target focal)")
    parser.add_argument("--use-multiscale-tta", action="store_true",
                        help="opt-in second-scale TTA (adds tta-second-scale px)")
    parser.add_argument("--tta-second-scale", type=int, default=456,
                        help="second TTA scale in px (default: %(default)s)")
    parser.add_argument("--no-tta", dest="use_tta", action="store_false",
                        help="disable test-time augmentation")
    args = parser.parse_args()

    # Auto-resume when a last checkpoint exists (crash recovery); start fresh otherwise.
    resume_path = 'efficientnetb4_last_checkpoint.pth'
    if not os.path.exists(resume_path):
        resume_path = None
    trainer = EfficientNetB4FinetuneTrainer(
        data_dir='HAM10000_split/train',
        num_epochs=args.num_epochs,
        resume_checkpoint=resume_path,
        use_metadata=args.use_metadata,
        metadata_csv=args.metadata_csv,
        lr_head=args.lr_head,
        stage2_unfreeze_patience=args.stage2_unfreeze_patience,
        use_ensemble=args.use_ensemble,
        keep_top_k_checkpoints=args.keep_top_k,
        use_tta=args.use_tta,
        use_mixup=args.use_mixup,
        use_cutmix=args.use_cutmix,
        use_multiscale_tta=args.use_multiscale_tta,
        tta_second_scale=args.tta_second_scale,
    )
    model, history, metrics = trainer.train()

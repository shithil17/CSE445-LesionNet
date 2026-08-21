"""
EXP-B4-CV-01 — 5-fold lesion-level cross-validation for the restored recipe.

Partitions the NON-TEST lesions (HAM10000_split/{train,val}, i.e. everything
outside the held-out test folder) into 5 class-stratified folds at the
lesion_id level, reusing split_ham10000.load_metadata() for the lesion->class /
lesion->images grouping (NOT duplicated here). Each fold trains one model with
the restored recipe (4 folds train, 1 fold val) via EfficientNetB4FinetuneTrainer
with per-fold working directories under Model/HAM10000_cv/fold{i}/ so
checkpoints/plots never clobber each other.

The held-out HAM10000_split/test folder is untouched by every fold during
training/selection and is used ONLY for the final report. Each fold's model is
temperature-scaled with its own T fitted on its own validation fold (TTA-averaged
logits, same as the single-split pipeline), and the 5 fold models' softmax
probabilities are averaged on the test set. The CV-ensemble Macro-F1 is reported
alongside the single-split baseline (from efficientnetb4_training_metrics.pth).

Gated behind an explicit flag on purpose: 5x training time. Run from Model/:

    ../.venv/bin/python cv_ham10000.py --cv-folds 5 [--num-epochs 50]
        [--use-metadata] [--use-mixup] [--use-cutmix] [--use-multiscale-tta]
"""

import argparse
import collections
import os
import random
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from EfficientNetB4_HAM10K import EfficientNetB4FinetuneTrainer, set_seed
from split_ham10000 import CLASSES, load_metadata

SPLIT = os.path.join(BASE, "HAM10000_split")
TEST_DIR = os.path.join(SPLIT, "test")
CV_ROOT = os.path.join(BASE, "HAM10000_cv")
DATA = os.path.dirname(BASE)
GROUNDTRUTH_CSV = os.path.join(DATA, "Dataset", "HAM10000_groundtruth.csv")
SEED = 42
DEFAULT_NUM_EPOCHS = 50


def make_folds(n_folds):
    """Lesion-level 5-fold partition, class-stratified, over the non-test
    lesions only. Mirrors split_ham10000.make_split(): per class, shuffle the
    lesion ids, then assign contiguous chunks to folds (no lesion crosses a
    fold boundary)."""
    lesion_class, lesion_images = load_metadata()
    image_to_lesion = {
        iid: lid for lid, imgs in lesion_images.items() for iid in imgs}
    used_lesions = set()
    for split_name in ("train", "val"):
        root = os.path.join(SPLIT, split_name)
        for cls in os.listdir(root):
            for fn in os.listdir(os.path.join(root, cls)):
                iid = os.path.splitext(fn)[0]
                used_lesions.add(image_to_lesion[iid])

    random.seed(SEED)
    folds = [set() for _ in range(n_folds)]
    for cls in CLASSES:
        lesions = sorted(l for l in used_lesions if lesion_class[l] == cls)
        random.shuffle(lesions)
        base = len(lesions) // n_folds
        extra = len(lesions) % n_folds
        start = 0
        for i in range(n_folds):
            size = base + (1 if i < extra else 0)
            folds[i].update(lesions[start:start + size])
            start += size
    return folds, image_to_lesion


def build_fold_trees(folds, image_to_lesion):
    """Symlink trees HAM10000_cv/fold{i}/{train,val}/{class}/*.jpg. A lesion is
    'val' in exactly one fold (its own), 'train' in the other four — so each
    fold sees 4/5 of the non-test lesions for training and a disjoint 1/5 for
    validation. Symlinks avoid duplicating ~10k images on disk."""
    if os.path.isdir(CV_ROOT):
        shutil.rmtree(CV_ROOT)
    for i in range(len(folds)):
        for split_name in ("train", "val"):
            out_root = os.path.join(CV_ROOT, f"fold{i}", split_name)
            for cls in CLASSES:
                os.makedirs(os.path.join(out_root, cls), exist_ok=True)
    for split_name in ("train", "val"):
        src_root = os.path.join(SPLIT, split_name)
        for cls in os.listdir(src_root):
            for fn in os.listdir(os.path.join(src_root, cls)):
                iid = os.path.splitext(fn)[0]
                lid = image_to_lesion[iid]
                for i, fold in enumerate(folds):
                    dest_split = "val" if lid in fold else "train"
                    dest = os.path.join(
                        CV_ROOT, f"fold{i}", dest_split, cls, fn)
                    os.symlink(os.path.join(src_root, cls, fn), dest)
    print(f"Fold trees built under {CV_ROOT} (symlinks)")


def run_fold(i, args):
    """Train one fold's model in its own working directory (checkpoint names,
    plots and metrics files are CWD-relative in the trainer). Returns the
    fold's trainer (holds class_names, metadata preprocessor state)."""
    fold_dir = os.path.join(CV_ROOT, f"fold{i}")
    work_dir = os.path.join(fold_dir, "work")
    os.makedirs(work_dir, exist_ok=True)
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        trainer = EfficientNetB4FinetuneTrainer(
            data_dir=os.path.join(fold_dir, "train"),
            val_data_dir=os.path.join(fold_dir, "val"),
            test_data_dir=TEST_DIR,
            experiment_id=f"EXP-B4-CV-FOLD{i}",
            num_epochs=args.num_epochs,
            seed=SEED,
            resume_checkpoint=None,
            use_metadata=args.use_metadata,
            metadata_csv=GROUNDTRUTH_CSV,
            lr_head=args.lr_head,
            use_ensemble=False,          # per-fold ensemble is meaningless; the
            keep_top_k_checkpoints=1,    # CV-ensemble happens across folds
            use_tta=args.use_tta,
            use_mixup=args.use_mixup,
            use_cutmix=args.use_cutmix,
            use_multiscale_tta=args.use_multiscale_tta,
            tta_second_scale=args.tta_second_scale,
        )
        model, history, metrics = trainer.train()
        return trainer
    finally:
        os.chdir(old_cwd)


def cv_ensemble_evaluation(trainers, use_tta=True):
    """Per-fold: reload the fold's best model, fit its own temperature on its
    own val fold (TTA-averaged logits), compute TTA'd temperature-scaled test
    probabilities. Average the 5 fold models' softmax probabilities on the
    final test set and report CV-ensemble metrics."""
    fold_probs = []
    all_targets = None
    fold_details = []
    for i, trainer in enumerate(trainers):
        fold_dir = os.path.join(CV_ROOT, f"fold{i}")
        work_dir = os.path.join(fold_dir, "work")
        best_path = os.path.join(work_dir, "efficientnetb4_best_model.pth")
        old_cwd = os.getcwd()
        os.chdir(work_dir)
        try:
            _, val_loader, test_loader = trainer.create_dataloaders()
            model = trainer.build_model()
            model.load_state_dict(
                torch.load(best_path, map_location="cpu", weights_only=False)[
                    "model_state_dict"])
            temperature = trainer.fit_temperature(model, val_loader)
            probs, targets, logits = trainer.predict_with_tta(model, test_loader)
            scaled = trainer._apply_temperature(logits, temperature)
            fold_probs.append(scaled)
            all_targets = targets
            preds = scaled.argmax(axis=1)
            fold_f1 = f1_score(targets, preds, average="macro", zero_division=0)
            fold_details.append((i, temperature, fold_f1))
            print(f"  fold {i}: T={temperature:.4f} | fold-model test Macro-F1 "
                  f"{fold_f1:.4f}")
        finally:
            os.chdir(old_cwd)

    print("\n" + "=" * 60)
    print("CV-ENSEMBLE (5 fold models, uniform softmax average)")
    print("=" * 60)
    for i, temperature, fold_f1 in fold_details:
        print(f"    fold {i}: T={temperature:.4f}, fold test Macro-F1={fold_f1:.4f}")
    avg_probs = np.mean(np.stack(fold_probs), axis=0)
    preds = avg_probs.argmax(axis=1)
    macro_f1 = f1_score(all_targets, preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(all_targets, preds)
    acc = 100.0 * float((preds == all_targets).mean())

    # ECE of the averaged (unscaled) and of the averaged ensemble probs.
    from EfficientNetB4_HAM10K import EfficientNetB4FinetuneTrainer as _T
    ece = _T._compute_ece(all_targets, preds, avg_probs)
    print(f"CV-ensemble Macro-F1   : {macro_f1:.4f}")
    print(f"CV-ensemble balanced acc: {balanced_acc:.4f}")
    print(f"CV-ensemble accuracy   : {acc:.2f}%")
    print(f"CV-ensemble ECE        : {ece:.4f}")
    print("=" * 60)

    baseline_path = os.path.join(BASE, "efficientnetb4_training_metrics.pth")
    if os.path.exists(baseline_path):
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
        base_f1 = baseline.get("tta_macro_f1")
        if base_f1 is not None:
            print(f"Single-split baseline (TTA) Macro-F1: {base_f1:.4f} | "
                  f"CV-ensemble delta: {macro_f1 - base_f1:+.4f}")
        else:
            print("Single-split baseline found but has no tta_macro_f1 key.")
    else:
        print("Single-split baseline not found "
              "(efficientnetb4_training_metrics.pth) — run the trainer first.")

    summary = {
        "experiment": "EXP-B4-CV-01",
        "cv_ensemble_macro_f1": macro_f1,
        "cv_ensemble_balanced_acc": balanced_acc,
        "cv_ensemble_accuracy": acc,
        "cv_ensemble_ece": ece,
        "fold_details": fold_details,
        "n_folds": len(fold_probs),
    }
    torch.save(summary, os.path.join(BASE, "cv_ensemble_metrics.pth"))
    print(f"Saved summary to Model/cv_ensemble_metrics.pth")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="5-fold lesion-level CV (restored recipe). Explicitly gated "
                    "behind --cv-folds: 5x training time.")
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="number of folds (must be >= 2; default 0 = disabled)")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--use-metadata", action="store_true")
    parser.add_argument("--use-mixup", action="store_true")
    parser.add_argument("--use-cutmix", action="store_true")
    parser.add_argument("--use-multiscale-tta", action="store_true")
    parser.add_argument("--tta-second-scale", type=int, default=456)
    parser.add_argument("--no-tta", dest="use_tta", action="store_false",
                        default=True)
    args = parser.parse_args()

    if args.cv_folds < 2:
        parser.error(
            "--cv-folds N (N >= 2) is required — CV is 5x training time and "
            "must be requested explicitly. Use --cv-folds 5 for the full run.")

    set_seed(SEED)
    folds, image_to_lesion = make_folds(args.cv_folds)
    build_fold_trees(folds, image_to_lesion)
    counts = [len(f) for f in folds]
    print(f"Fold val-lesion counts: {counts} (total non-test lesions "
          f"{sum(counts)})")

    trainers = []
    for i in range(args.cv_folds):
        print(f"\n{'=' * 60}\nTRAINING FOLD {i}/{args.cv_folds}\n{'=' * 60}")
        trainers.append(run_fold(i, args))

    cv_ensemble_evaluation(trainers, use_tta=args.use_tta)


if __name__ == "__main__":
    main()
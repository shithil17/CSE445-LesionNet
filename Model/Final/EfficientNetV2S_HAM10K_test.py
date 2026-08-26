"""
Standalone test harness for trained EfficientNetV2-S checkpoints.

Reuses the trainer from EfficientNetV2S_HAM10K.py (same transforms, TTA,
temperature scaling, ensemble and metric code) but every plot is saved as a
seaborn-styled PDF into Model/Final/outputs/.

Run from the repository root:
    .venv/bin/python 'Model/Final/EfficientNetV2S_HAM10K_test.py'
    .venv/bin/python 'Model/Final/EfficientNetV2S_HAM10K_test.py' \
        --checkpoint efficientnetv2s_best_accuracy_model.pth --tag acc-best
    .venv/bin/python 'Model/Final/EfficientNetV2S_HAM10K_test.py' \
        --checkpoint efficientnetv2s_best_model_epoch12.pth \
        --checkpoint efficientnetv2s_best_model_epoch19.pth \
        --checkpoint efficientnetv2s_best_model_epoch22.pth --tag ensemble
    .venv/bin/python 'Model/Final/EfficientNetV2S_HAM10K_test.py' \
        --temperature 0.6 --no-tta --split val
    .venv/bin/python 'Model/Final/EfficientNetV2S_HAM10K_test.py' \
        --train-curves efficientnetv2s_training_metrics.pth
"""

import argparse
import contextlib
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import f1_score

from EfficientNetV2S_HAM10K import (
    CHECKPOINT_PREFIX,
    DEFAULT_TEST_DATA_DIR,
    DEFAULT_TRAIN_DATA_DIR,
    DEFAULT_VAL_DATA_DIR,
    EfficientNetFinetuneTrainer,
    set_seed,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "outputs")


class PDFTestTrainer(EfficientNetFinetuneTrainer):
    """Same trainer, but every plot method saves a PDF into outputs/."""

    def __init__(self, out_dir, tag, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dir = out_dir
        self.tag = tag

    @contextlib.contextmanager
    def _pdf_output(self, name):
        real_savefig = plt.savefig

        def savefig(path, *args, **kwargs):
            return real_savefig(
                os.path.join(self.out_dir, f"{self.tag}_{name}.pdf"),
                *args, **kwargs)

        plt.savefig = savefig
        try:
            yield
        finally:
            plt.savefig = real_savefig

    def plot_roc_curves(self, y_true, probabilities):
        with self._pdf_output("roc_curves"):
            return super().plot_roc_curves(y_true, probabilities)

    def plot_pr_curves(self, y_true, probabilities):
        with self._pdf_output("precision_recall_curves"):
            return super().plot_pr_curves(y_true, probabilities)

    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        with self._pdf_output(f"confusion_matrix_epoch{epoch}"):
            return super().plot_confusion_matrix(y_true, y_pred, epoch)

    def plot_final_confusion_matrix(self, y_true, y_pred):
        with self._pdf_output("confusion_matrix_final"):
            return super().plot_final_confusion_matrix(y_true, y_pred)

    def plot_confidence_distribution(self, confidence, correct):
        with self._pdf_output("confidence_distribution"):
            return super().plot_confidence_distribution(confidence, correct)

    def plot_calibration_curve(self, y_true, y_pred, probabilities, n_bins=10):
        with self._pdf_output("calibration_curve"):
            return super().plot_calibration_curve(
                y_true, y_pred, probabilities, n_bins=n_bins)

    def _unfreeze_lines(self, ax):
        ax.axvline(x=6, linestyle='--', color='gray',
                   label=f'Unfreeze {self._early_blocks_label} (epoch 6)')
        if self._stage2_epoch is not None:
            ax.axvline(x=self._stage2_epoch, linestyle='--', color='orange',
                       label=f'Unfreeze {self._stage2_blocks_label} '
                             f'(epoch {self._stage2_epoch})')

    def plot_training_curves(self, history):
        """One PDF per curve type (acc, loss, lr, val acc+F1). Macro-F1 is
        plotted x100 (percent) so it shares the accuracy axis cleanly."""
        epochs = range(1, len(history['train_loss']) + 1)

        with self._pdf_output("training_curves_acc"):
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
            plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
            self._unfreeze_lines(plt.gca())
            plt.title('Training & Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("_", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training accuracy curve saved as PDF in '{self.out_dir}/'")

        with self._pdf_output("training_curves_loss"):
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
            plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            self._unfreeze_lines(plt.gca())
            plt.title('Training & Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("_", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training loss curve saved as PDF in '{self.out_dir}/'")

        if 'lr' in history:
            with self._pdf_output("training_curves_lr"):
                plt.figure(figsize=(8, 6))
                plt.plot(epochs, history['lr'], 'g-')
                plt.title('Learning Rate (classifier group)')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("_", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Learning rate curve saved as PDF in '{self.out_dir}/'")

        with self._pdf_output("training_curves_val_acc_f1"):
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
            plt.plot(epochs, np.array(history['val_macro_f1']) * 100, 'm-',
                     label='Validation Macro-F1 (x100)')
            self._unfreeze_lines(plt.gca())
            plt.title('Validation Accuracy & Macro-F1')
            plt.xlabel('Epoch')
            plt.ylabel('Score (%)')
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("_", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Validation accuracy/F1 curve saved as PDF in '{self.out_dir}/'")

    def plot_trainable_parameters(self, history):
        with self._pdf_output("trainable_parameters"):
            return super().plot_trainable_parameters(history)

    def plot_generalization_gap(self, history):
        """One PDF per gap (accuracy, loss)."""
        epochs = range(1, len(history['train_loss']) + 1)

        with self._pdf_output("generalization_gap_acc"):
            plt.figure(figsize=(8, 6))
            accuracy_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
            plt.plot(epochs, accuracy_gap, 'b-o', label='Accuracy Gap (train - val)')
            plt.title('Accuracy Generalization Gap')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy Gap (%)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("_", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Accuracy gap curve saved as PDF in '{self.out_dir}/'")

        with self._pdf_output("generalization_gap_loss"):
            plt.figure(figsize=(8, 6))
            loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
            plt.plot(epochs, loss_gap, 'r-o', label='Loss Gap (val - train)')
            plt.title('Loss Generalization Gap')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Gap')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("_", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Loss gap curve saved as PDF in '{self.out_dir}/'")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained EfficientNetV2-S checkpoints (single or "
                    "F1-weighted ensemble) and export all plots as PDFs to "
                    "Model/Final/outputs/.")
    parser.add_argument("--checkpoint", action="append", metavar="PATH",
                        help="checkpoint .pth to test; repeat the flag for an "
                             "F1-weighted ensemble (default: best_model.pth)")
    parser.add_argument("--tag", default=None,
                        help="output filename prefix (default: checkpoint stem)")
    parser.add_argument("--no-tta", dest="use_tta", action="store_false",
                        default=True, help="disable test-time augmentation")
    parser.add_argument("--temperature", default="auto",
                        help="auto (fit on validation) | none (T=1.0) | FLOAT")
    parser.add_argument("--split", choices=("test", "val"), default="test",
                        help="split to report on (temperature is always fit "
                             "on validation)")
    parser.add_argument("--train-curves", metavar="METRICS_PTH", default=None,
                        help="also export train/val curves from a "
                             "training_metrics.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.chdir(EXPERIMENT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    checkpoints = args.checkpoint or [f"{CHECKPOINT_PREFIX}_best_model.pth"]
    for path in checkpoints:
        if not os.path.exists(path):
            parser.error(f"checkpoint not found: {path}")

    tag = args.tag or "+".join(
        os.path.splitext(os.path.basename(p))[0] for p in checkpoints)

    trainer = PDFTestTrainer(
        OUTPUT_DIR, tag,
        data_dir=DEFAULT_TRAIN_DATA_DIR,
        val_data_dir=DEFAULT_VAL_DATA_DIR,
        test_data_dir=DEFAULT_TEST_DATA_DIR,
        use_tta=args.use_tta,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = trainer.create_dataloaders()
    model = trainer.build_model()
    criterion = trainer._build_criterion()
    split_loader = test_loader if args.split == "test" else val_loader

    checkpoints_loaded = []
    for path in checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("backbone", "v2s") != trainer.backbone:
            raise ValueError(
                f"checkpoint {path} has backbone={ckpt.get('backbone')!r}, "
                f"expected {trainer.backbone!r}")
        checkpoints_loaded.append(ckpt)
    model.load_state_dict(checkpoints_loaded[0]["model_state_dict"])

    no_tta_f1 = None
    if args.use_tta:
        _, no_tta_acc, no_tta_preds, no_tta_targets, _ = trainer.validate(
            model, split_loader, criterion, epoch_label=args.split)
        no_tta_f1 = f1_score(no_tta_preds, no_tta_targets, average="macro",
                             zero_division=0)

    if len(checkpoints) == 1:
        if args.use_tta:
            _, val_targets_single, _ = trainer.predict_with_tta(
                model, val_loader)
            test_probs, test_targets, test_logits = trainer.predict_with_tta(
                model, split_loader)
        else:
            _, _, _, val_targets_single, _ = trainer.validate(
                model, val_loader, criterion, epoch_label="val")
            _, _, _, test_targets, test_probs = trainer.validate(
                model, split_loader, criterion, epoch_label=args.split)
            test_logits, _ = trainer.collect_logits(
                model, split_loader, epoch_label=args.split)

        if args.temperature == "auto":
            temperature = trainer.fit_temperature(model, val_loader)
            fallback = trainer._temperature_fallback_single
            source = trainer._temperature_source_single
        elif args.temperature == "none":
            temperature, fallback, source = 1.0, True, "T=1.0 (user)"
        else:
            temperature, fallback, source = float(args.temperature), False, "user"

        scaled = trainer._apply_temperature(test_logits, temperature)
        model_label = f"single {os.path.basename(checkpoints[0])}" + \
            (" (TTA)" if args.use_tta else " (no TTA)")
        best_epoch = checkpoints_loaded[0].get("epoch", 1)
    else:
        members = [
            {"path": p, "f1": float(c.get("best_macro_f1", 0.0)),
             "epoch": c.get("epoch", 0)}
            for p, c in zip(checkpoints, checkpoints_loaded)
        ]
        val_probs, val_targets_single, _ = trainer._ensemble_probs(
            model, val_loader, members, criterion, use_tta=args.use_tta)
        test_probs, test_targets, _ = trainer._ensemble_probs(
            model, split_loader, members, criterion, use_tta=args.use_tta)

        if args.temperature == "auto":
            temperature = trainer.fit_temperature_on_probs(
                val_probs, val_targets_single)
            fallback = trainer._temperature_fallback_ens
            source = trainer._temperature_source_ens
        elif args.temperature == "none":
            temperature, fallback, source = 1.0, True, "T=1.0 (user)"
        else:
            temperature, fallback, source = float(args.temperature), False, "user"

        scaled = trainer._apply_temperature(
            np.log(np.clip(test_probs, 1e-12, 1.0)), temperature)
        model_label = f"ensemble ({len(members)} members)" + \
            (" (TTA)" if args.use_tta else " (no TTA)")
        best_epoch = max(m["epoch"] for m in members)

    final_preds = scaled.argmax(axis=1)
    final_acc = 100.0 * float((final_preds == test_targets).mean())
    ece_before = trainer._compute_ece(
        test_targets, test_probs.argmax(axis=1), test_probs)
    ece_after = trainer._compute_ece(test_targets, final_preds, scaled)
    trainer._check_temperature_invariance(test_probs, scaled)

    if args.use_tta:
        tta_f1 = f1_score(final_preds, test_targets, average="macro",
                          zero_division=0)
        print(f"\n{args.split} Macro-F1: no-TTA {no_tta_f1:.4f} -> "
              f"final {tta_f1:.4f}")

    report = trainer._final_metrics_report(
        test_targets, final_preds, scaled, final_acc, best_epoch, ece_after,
        model_label, ece_before=ece_before, temperature=temperature,
        argmax_invariant=True, temperature_fallback=fallback,
        temperature_source=source)

    if args.train_curves:
        metrics = torch.load(args.train_curves, map_location="cpu",
                             weights_only=False)
        trainer._stage2_epoch = metrics.get("stage2_epoch")
        trainer.plot_training_curves(metrics["history"])
        trainer.plot_trainable_parameters(metrics["history"])
        trainer.plot_generalization_gap(metrics["history"])

    out = {
        "experiment_id": checkpoints_loaded[0].get("experiment_id"),
        "backbone": trainer.backbone,
        "checkpoints": checkpoints,
        "split": args.split,
        "use_tta": args.use_tta,
        "use_mixup": trainer.use_mixup,
        "temperature": temperature,
        "temperature_fallback": fallback,
        "temperature_source": source,
        "no_tta_macro_f1": no_tta_f1,
        "final_accuracy": final_acc,
        "class_names": trainer.class_names,
        "class_counts": dict(zip(trainer.class_names, trainer.class_counts)),
        "split_sizes": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
    }
    out.update(report)
    metrics_path = os.path.join(OUTPUT_DIR, f"{tag}_metrics.pth")
    torch.save(out, metrics_path)
    print(f"\nMetrics saved to '{metrics_path}'")
    print(f"Plots exported to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
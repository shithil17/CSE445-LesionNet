import os
import shutil
import random
import warnings
from typing import Dict, List

import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# NOTE: Not part of the active training pipeline — EfficientNetB4_HAM10K.py trains on the raw
# HAM10000_split/train with on-the-fly augmentation. This offline balanced-dataset builder is
# retained for reference/experimentation only.


class HAM10000AugmentorFromTable:
    def __init__(
        self,
        image_dir: str,
        output_dir: str,
        seed: int = 42,
    ):
        """
        image_dir: class-organized directory tree, image_dir/<CLASS>/*.jpg
            (CLASS in AKIEC, BCC, BKL, DF, MEL, NV, VASC)
        output_dir: where to write output_dir/<CLASS>/images.jpg (+ aug)
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.class_names = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

        os.makedirs(self.output_dir, exist_ok=True)
        for c in self.class_names:
            os.makedirs(os.path.join(self.output_dir, c), exist_ok=True)

        # Seed Python's random, numpy, and torch: torchvision's RandomRotation /
        # RandomAffine / RandomResizedCrop draw from torch's RNG internally, so seeding
        # only `random` does not make the augmented output reproducible.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # SKINC-NET target counts (Table 2 as per your script)
        self.target_counts = {
            "NV": 6705,
            "MEL": 6678,
            "BKL": 6594,
            "BCC": 6682,
            "AKIEC": 6213,
            "VASC": 6674,
            "DF": 6555,
        }

        # Augmentation pipeline
        self.augment_transform = transforms.Compose(
            [
                transforms.RandomRotation(20),
                transforms.RandomAffine(
                    degrees=0,
                    shear=(-0.2, 0.2),  # radians-like proportion
                    fill=0,
                ),
                transforms.RandomResizedCrop(
                    size=(224, 224), scale=(0.9, 1.1), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

        # Index images by scanning the class folders (ImageFolder-style)
        self.class_to_images = self._scan_classes()

    def _scan_classes(self) -> Dict[str, List[str]]:
        class_to_images: Dict[str, List[str]] = {}
        for c in self.class_names:
            cdir = os.path.join(self.image_dir, c)
            if not os.path.isdir(cdir):
                warnings.warn(f"Class folder not found: {cdir}; skipping")
                class_to_images[c] = []
                continue
            class_to_images[c] = sorted(
                f for f in os.listdir(cdir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
        return class_to_images

    def get_class_distribution(self) -> Dict[str, int]:
        return {c: len(self.class_to_images.get(c, [])) for c in self.class_names}

    def calculate_augmentation_factors(self):
        dist = self.get_class_distribution()
        factors = {}
        print("Current dataset distribution:")
        for c in self.class_names:
            original = dist.get(c, 0)
            target = self.target_counts[c]
            need = max(0, target - original)
            factor = (target - original) / original if original > 0 and original < target else 0
            print(f"  {c}: {original} -> {target} (need {need})")
            factors[c] = {
                "original": original,
                "target": target,
                "augmented_needed": need,
                "factor": factor,
            }
        return factors

    def _copy_originals(self, class_name: str, files: List[str]):
        src_dir = os.path.join(self.image_dir, class_name)
        out_dir = os.path.join(self.output_dir, class_name)
        copied = 0
        for fname in files:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(out_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
        return copied

    def _augment_to_target(self, class_name: str, original_files: List[str], need: int):
        if need <= 0:
            return 0

        src_dir = os.path.join(self.image_dir, class_name)
        out_dir = os.path.join(self.output_dir, class_name)
        augmented = 0

        # cycle through originals until we hit 'need'
        idx = 0
        n = len(original_files)
        if n == 0:
            warnings.warn(f"No originals for {class_name}; cannot augment.")
            return 0

        while augmented < need:
            fname = original_files[idx % n]
            src_path = os.path.join(src_dir, fname)
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                warnings.warn(f"Failed to open {src_path}: {e}")
                idx += 1
                continue

            aug_img = self.augment_transform(img)
            tensor = self.to_tensor(aug_img)
            pil_img = self.to_pil(tensor)

            base = os.path.splitext(os.path.basename(fname))[0]
            save_name = f"{base}_aug_{augmented}.jpg"
            save_path = os.path.join(out_dir, save_name)
            pil_img.save(save_path, quality=95)
            augmented += 1
            idx += 1

        return augmented

    def run(self):
        print("Starting HAM10000 augmentation from split folder...")
        print("=" * 60)
        factors = self.calculate_augmentation_factors()
        print("=" * 60)

        for c in self.class_names:
            originals = self.class_to_images.get(c, [])
            print(f"\nProcessing {c}:")
            copied = self._copy_originals(c, originals)
            print(f"  Copied {copied} originals (existing files skipped)")

            need = factors[c]["augmented_needed"]
            if need > 0:
                print(f"  Generating {need} augmented images...")
                made = self._augment_to_target(c, originals, need)
                print(f"  {c}: Generated {made} augmented images")
            else:
                print("  No augmentation needed (at or above target).")

        print("\n" + "=" * 60)
        print("Augmentation complete! Final distribution:")
        print("=" * 60)

        grand = 0
        for c in self.class_names:
            out_c = os.path.join(self.output_dir, c)
            count = len(
                [
                    f
                    for f in os.listdir(out_c)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            grand += count
            print(f"  {c}: {count}")
        print(f"\nTotal dataset size: {grand:,} images")
        print(f"Saved to: {self.output_dir}")


if __name__ == "__main__":
    augmentor = HAM10000AugmentorFromTable(
        image_dir="HAM10000_split/train",
        output_dir="HAM10K_augmented_dataset",
    )
    augmentor.run()
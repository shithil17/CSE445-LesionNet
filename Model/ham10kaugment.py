import os
import csv
import math
import shutil
import random
import warnings
from typing import Dict, List, Tuple, Optional

import torch
from torchvision import transforms
from PIL import Image

try:
    import pandas as pd  # for xlsx/csv flexibility
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


class HAM10000AugmentorFromTable:
    def __init__(
        self,
        image_dir: str,
        labels_path: str,
        output_dir: str,
        seed: int = 42,
    ):
        """
        image_dir: directory containing all .jpg dermoscopy images (flat)
        labels_path: CSV or XLSX with columns:
            image, MEL, NV, BCC, AKIEC, BKL, DF, VASC (one-hot)
        output_dir: where to write output_dir/<CLASS>/images.jpg (+ aug)
        """
        self.image_dir = image_dir
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.class_names = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

        os.makedirs(self.output_dir, exist_ok=True)
        for c in self.class_names:
            os.makedirs(os.path.join(self.output_dir, c), exist_ok=True)

        random.seed(seed)

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

        # Load label table and image indexes per class
        self.image_to_class, self.class_to_images = self._load_labels()

    def _read_table(self):
        ext = os.path.splitext(self.labels_path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            if not HAS_PANDAS:
                raise RuntimeError(
                    "pandas is required to read Excel. Install pandas or convert to CSV."
                )
            df = pd.read_excel(self.labels_path)
            return df
        elif ext == ".csv":
            if HAS_PANDAS:
                return pd.read_csv(self.labels_path)
            # Fallback csv reader
            with open(self.labels_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # Convert to a pandas-like minimal interface
            if not HAS_PANDAS:
                # Create a simple structure
                return rows
        else:
            raise ValueError("labels_path must be .csv or .xlsx")

    def _normalize_rows(self, rows):
        # Coerce to list[dict] with required columns
        required = ["image", "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        if HAS_PANDAS and hasattr(rows, "to_dict"):
            rows = rows.to_dict(orient="records")
        # Lowercase headers in rows; accept numeric 0/1 or string "0"/"1"
        norm = []
        for r in rows:
            r_norm = {}
            for k, v in r.items():
                r_norm[k.strip()] = v
            # Validate required columns exist
            for col in required:
                if col not in r_norm:
                    raise KeyError(f"Missing column '{col}' in labels file")
            norm.append(r_norm)
        return norm

    def _row_to_class(self, row: dict) -> Optional[str]:
        # HAM10000 is single-label; if multiple 1s, pick the first by class order
        hits = []
        for c in self.class_names:
            val = row.get(c, 0)
            try:
                v = int(val)
            except Exception:
                try:
                    v = int(float(val))
                except Exception:
                    v = 0
            if v == 1:
                hits.append(c)
        if len(hits) == 0:
            return None
        if len(hits) > 1:
            warnings.warn(
                f"Row for {row.get('image')} has multiple labels {hits}; "
                f"using {hits[0]}"
            )
        return hits[0]

    def _load_labels(self) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        rows = self._read_table()
        rows = self._normalize_rows(rows)

        image_to_class: Dict[str, str] = {}
        class_to_images: Dict[str, List[str]] = {c: [] for c in self.class_names}

        for r in rows:
            image_key = str(r["image"]).strip()
            # allow bare IDs or filenames; add .jpg if needed
            if not image_key.lower().endswith((".jpg", ".jpeg", ".png")):
                candidate_jpg = image_key + ".jpg"
            else:
                candidate_jpg = image_key

            label = self._row_to_class(r)
            if label is None:
                continue

            # ensure file exists
            src_path = os.path.join(self.image_dir, candidate_jpg)
            if not os.path.exists(src_path):
                # try .jpeg/.png fallbacks
                alt = None
                for ext in [".jpeg", ".png"]:
                    p = os.path.join(self.image_dir, image_key + ext)
                    if os.path.exists(p):
                        alt = p
                        candidate_jpg = image_key + ext
                        break
                if alt is None:
                    warnings.warn(f"Image not found: {candidate_jpg}; skipping")
                    continue

            image_to_class[candidate_jpg] = label
            class_to_images[label].append(candidate_jpg)

        return image_to_class, class_to_images

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
            print(f"  {c}: {original} → {target} (need {need})")
            factors[c] = {
                "original": original,
                "target": target,
                "augmented_needed": need,
                "factor": factor,
            }
        return factors

    def _copy_originals(self, class_name: str, files: List[str]):
        out_dir = os.path.join(self.output_dir, class_name)
        copied = 0
        for fname in files:
            src = os.path.join(self.image_dir, fname)
            dst = os.path.join(out_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
        return copied

    def _augment_to_target(self, class_name: str, original_files: List[str], need: int):
        if need <= 0:
            return 0

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
            src_path = os.path.join(self.image_dir, fname)
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
        print("Starting HAM10000 augmentation from table...")
        print("=" * 60)
        factors = self.calculate_augmentation_factors()
        print("=" * 60)

        total_final = 0
        for c in self.class_names:
            originals = self.class_to_images.get(c, [])
            print(f"\nProcessing {c}:")
            copied = self._copy_originals(c, originals)
            print(f"  Copied {copied} originals (existing files skipped)")

            # NV is majority; still copy originals but no augmentation beyond target logic
            need = factors[c]["augmented_needed"]
            if need > 0 and c != "NV":
                print(f"  Generating {need} augmented images...")
                made = self._augment_to_target(c, originals, need)
                print(f"  {c}: Generated {made} augmented images")
            else:
                if need <= 0:
                    print("  No augmentation needed (at or above target).")
                else:
                    print("  NV is majority class; skipping augmentation.")

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
        image_dir="../Dataset/HAM10000_images",
        labels_path="../Dataset/HAM10000_groundtruth.csv",  # or .xlsx
        output_dir="HAM10K_augmented_dataset",
    )
    augmentor.run()
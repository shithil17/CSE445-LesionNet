import os
import csv
import random
import shutil
import collections
from typing import Dict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(BASE), "Dataset")
SRC_IMAGES = os.path.join(DATA, "HAM10000_images")
META = os.path.join(DATA, "HAM10000_metadata.csv")
OUT = os.path.join(BASE, "HAM10000_split")

CLASSES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
RATIOS = {"train": 0.72, "val": 0.08, "test": 0.20}
SEED = 42


def load_metadata():
    lesion_class: Dict[str, str] = {}
    lesion_images = collections.defaultdict(list)
    with open(META) as f:
        for r in csv.DictReader(f):
            dx = r["dx"].upper()
            if dx not in CLASSES:
                raise SystemExit(f"Unknown dx '{r['dx']}' for {r['image_id']}")
            lesion_class[r["lesion_id"]] = dx
            lesion_images[r["lesion_id"]].append(r["image_id"])
    return lesion_class, lesion_images


def make_split():
    """Assign every lesion (not image) to train/val/test, stratified per class."""
    lesion_class, lesion_images = load_metadata()
    random.seed(SEED)

    assign: Dict[str, str] = {}
    for cls in CLASSES:
        lesions = sorted(l for l, c in lesion_class.items() if c == cls)
        random.shuffle(lesions)
        n_train = int(round(len(lesions) * RATIOS["train"]))
        n_val = int(round(len(lesions) * RATIOS["val"]))
        for i, lid in enumerate(lesions):
            if i < n_train:
                assign[lid] = "train"
            elif i < n_train + n_val:
                assign[lid] = "val"
            else:
                assign[lid] = "test"

    return lesion_class, lesion_images, assign


def main():
    lesion_class, lesion_images, assign = make_split()

    img = {s: collections.Counter() for s in RATIOS}
    les = {s: collections.Counter() for s in RATIOS}

    # Rebuild the split tree from scratch (raw HAM10000_images untouched)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)

    lesion_ids = set(assign)
    for lid, imgs in lesion_images.items():
        split = assign[lid]
        cls = lesion_class[lid]
        out_dir = os.path.join(OUT, split, cls)
        os.makedirs(out_dir, exist_ok=True)
        for iid in imgs:
            src = os.path.join(SRC_IMAGES, iid + ".jpg")
            if not os.path.exists(src):
                raise SystemExit(f"Missing source image: {src}")
            shutil.copy2(src, os.path.join(out_dir, iid + ".jpg"))
        img[split][cls] += len(imgs)
        les[split][cls] += 1

    print("=" * 72)
    print("HAM10000 lesion-level 3-way split (no lesion crosses a boundary)")
    print(f"Source: {SRC_IMAGES}")
    print(f"Output: {OUT}")
    print("=" * 72)
    print(f"{'Class':<8} {'Tr img':>7} {'Va img':>7} {'Te img':>7} "
          f"{'Tr les':>7} {'Va les':>7} {'Te les':>7}")
    print("-" * 72)
    for c in CLASSES:
        print(f"{c:<8} {img['train'][c]:>7} {img['val'][c]:>7} {img['test'][c]:>7} "
              f"{les['train'][c]:>7} {les['val'][c]:>7} {les['test'][c]:>7}")
    print("-" * 72)
    print(f"{'TOTAL':<8} {sum(img['train'].values()):>7} {sum(img['val'].values()):>7} "
          f"{sum(img['test'].values()):>7} {sum(les['train'].values()):>7} "
          f"{sum(les['val'].values()):>7} {sum(les['test'].values()):>7}")
    print("=" * 72)

    total_img = sum(sum(img[s].values()) for s in RATIOS)
    total_les = sum(sum(les[s].values()) for s in RATIOS)
    assert sum(len(v) for v in lesion_images.values()) == 10015, "lesion->image mapping broken"
    print(f"Total images copied: {total_img:,} (expect 10,015)")
    print(f"Total lesions: {total_les:,}")
    assert total_img == 10_015 and total_les == len(lesion_ids) == len(lesion_images)

    print("\nSplit complete. (HAM10000_images untouched — files were copied, not moved.)")


if __name__ == "__main__":
    main()
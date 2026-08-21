"""GT-03-METADATA-CORRECTION: HAM10000 ground-truth metadata cleaning fix.

Fixes the categorical metadata semantics of HAM10000_groundtruth.csv:

  * 'unknown' is NO LONGER a synthetic fallback for missing/unrecognized values.
    The original source (HAM10000_metadata.csv) LITERALLY contains 'unknown' as
    a non-missing category (57 sex rows, 234 localization rows) — those are
    preserved as genuine categories with *_missing = 0. All other values are
    preserved exactly (strip + lowercase only).
  * A genuinely missing source value is written as an EMPTY STRING with
    *_missing = 1. Missing and 'unknown' stay distinct.
  * Any non-empty sex value outside the source-observed canonical set
    (male, female, unknown) fails the build loudly — no silent mutation.
  * Age handling is unchanged (empty -> "" + age_missing=1; no imputation).
  * The seven target one-hot columns are byte-identical in semantics.

After writing, the output CSV is RELOADED independently and compared row-by-row
against the original source (image_id / image join): image id, lesion_id,
diagnosis, age, sex, localization must all be preserved up to explicit missing
representation and normalization. Any semantic drift fails the build.

Experiment ID : GT-03-METADATA-CORRECTION
Schema        : v3

Source : ../Dataset/HAM10000_metadata.csv
Output : ../Dataset/HAM10000_groundtruth.csv

Run from the Model/ directory:
    ../.venv/bin/python build_groundtruth.py
"""

import csv
import os
from collections import Counter

META = "../Dataset/HAM10000_metadata.csv"
OUT = "../Dataset/HAM10000_groundtruth.csv"

EXPERIMENT_ID = "GT-03-METADATA-CORRECTION"
SCHEMA_VERSION = "v3"

# ------------------------------------------------------------------ column roles
# Targets: exactly one-hot 1 per row (ground truth, NOT model inputs).
CLASS_ORDER = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
TARGET_COLUMNS = list(CLASS_ORDER)

# Identifiers: join keys for the lesion-level split and provenance (NOT inputs).
IDENTIFIER_COLUMNS = ["image", "lesion_id"]

# Model features for the FUTURE image + metadata experiment only.
# Raw/categorical values; normalization/encoding is deferred to the model side.
METADATA_FEATURE_COLUMNS = [
    "age",
    "age_missing",
    "sex",
    "sex_missing",
    "localization",
    "localization_missing",
]

COLUMNS = IDENTIFIER_COLUMNS + METADATA_FEATURE_COLUMNS + TARGET_COLUMNS

# ------------------------------------------------------------------ source rules
REQUIRED_SOURCE_COLUMNS = ["image_id", "lesion_id", "dx", "age", "sex", "localization"]

DX_TO_CLASS = {c.lower(): c for c in CLASS_ORDER}

# Canonical non-empty sex values. 'unknown' is NOT a fallback: it is a literal
# non-missing category in the original HAM10000_metadata.csv (57 rows) and must
# be preserved as-is. Any other non-empty value is a data error -> loud failure.
SEX_CANONICAL = {"male", "female", "unknown"}

# ------------------------------------------------------------------ cleaning


def _parse_age(raw, image_id):
    """Return (value_or_None, missing_flag). Empty -> missing (no imputation).
    Non-empty non-numeric -> hard error, never silently marked missing."""
    raw = raw.strip()
    if not raw:
        return None, 1
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"Unparseable age '{raw}' for image {image_id}")
    if value < 0:
        raise ValueError(f"Negative age '{raw}' for image {image_id}")
    return value, 0


def _normalize_sex(raw, image_id):
    """Strip/lowercase. Empty -> missing ("", 1). Source-observed canonical
    category -> preserved with flag 0. Anything else -> LOUD failure."""
    value = raw.strip().lower()
    if value == "":
        return "", 1
    if value in SEX_CANONICAL:
        return value, 0
    raise ValueError(f"Unexpected sex value '{raw}' for image {image_id}")


def _normalize_localization(raw):
    """Strip/lowercase, preserve the semantic category exactly. Empty -> missing
    ("", 1). Non-empty (including 'unknown') -> preserved with flag 0.
    No category merging, no synthetic 'unknown'."""
    value = raw.strip().lower()
    if value == "":
        return "", 1
    return value, 0


def _collect_source_observations(source_rows):
    """Sorted non-empty raw values as observed in the ORIGINAL source (audit)."""
    sex = sorted({r["sex"].strip().lower() for r in source_rows if r["sex"].strip()})
    loc = sorted({r["localization"].strip().lower() for r in source_rows
                  if r["localization"].strip()})
    return sex, loc


# ------------------------------------------------------------------ validation


def _validate_strict(rows, source_rows):
    """§12: structural + semantic invariants; every violation fails the build."""
    if len(rows) != len(source_rows):
        raise ValueError(
            f"Row count mismatch: {len(rows)} output vs {len(source_rows)} source"
        )
    seen_images = set()
    for row in rows:
        if row["image"] in seen_images:
            raise ValueError(f"Duplicate output image: {row['image']}")
        seen_images.add(row["image"])

        # Every image has exactly one lesion_id (join key for the split).
        if not row["lesion_id"]:
            raise ValueError(f"Empty lesion_id for image {row['image']}")

        # Sex: exactly one of (canonical category, flag 0) or ("", flag 1).
        sex_ok = (
            (row["sex"] in SEX_CANONICAL and row["sex_missing"] == 0)
            or (row["sex"] == "" and row["sex_missing"] == 1)
        )
        if not sex_ok:
            raise ValueError(
                f"Invalid sex representation for image {row['image']}: "
                f"sex={row['sex']!r} sex_missing={row['sex_missing']}"
            )

        # Localization: (non-empty, flag 0) or ("", flag 1).
        loc_ok = (
            (row["localization"] != "" and row["localization_missing"] == 0)
            or (row["localization"] == "" and row["localization_missing"] == 1)
        )
        if not loc_ok:
            raise ValueError(
                f"Invalid localization representation for image {row['image']}: "
                f"localization={row['localization']!r} "
                f"localization_missing={row['localization_missing']}"
            )

        # Age: numeric or explicit missing.
        if row["age_missing"] == 0 and row["age"] == "":
            raise ValueError(f"Empty age without missing flag for image {row['image']}")
        if row["age_missing"] == 1 and row["age"] != "":
            raise ValueError(f"Non-empty age with missing flag for image {row['image']}")

        # Targets: exactly one 1.
        if sum(row[c] for c in TARGET_COLUMNS) != 1:
            raise ValueError(
                f"One-hot invariant violated for image {row['image']}"
            )


def _verify_against_source(out_path):
    """§13: independently reload the ORIGINAL source and the written ground
    truth, join row-by-row on image_id/image, and prove semantic preservation."""
    with open(META, newline="", encoding="utf-8") as f:
        source = {r["image_id"]: r for r in csv.DictReader(f)}
    with open(out_path, newline="", encoding="utf-8") as f:
        out = {r["image"]: r for r in csv.DictReader(f)}

    if set(source) != set(out):
        raise ValueError("Image id sets differ between source and ground truth")

    for image_id, s in source.items():
        o = out[image_id]

        # Identifiers unchanged.
        if o["lesion_id"] != s["lesion_id"].strip():
            raise ValueError(f"lesion_id changed for image {image_id}")

        # Diagnosis unchanged.
        if DX_TO_CLASS[s["dx"].strip().lower()] != next(
            c for c in TARGET_COLUMNS if o[c] == "1"
        ):
            raise ValueError(f"Diagnosis changed for image {image_id}")

        # Age: unchanged up to explicit missing representation.
        if s["age"].strip() == "":
            if o["age"] != "" or o["age_missing"] != "1":
                raise ValueError(f"Missing age representation changed for image {image_id}")
        else:
            if abs(float(o["age"]) - float(s["age"])) > 1e-9 or o["age_missing"] != "0":
                raise ValueError(f"Age value changed for image {image_id}")

        # Sex: unchanged up to strip/lowercase; flag reflects source emptiness.
        expected_sex = s["sex"].strip().lower()
        expected_flag = "0" if s["sex"].strip() else "1"
        if o["sex"] != expected_sex or o["sex_missing"] != expected_flag:
            raise ValueError(
                f"Sex changed for image {image_id}: source={s['sex']!r} "
                f"out={o['sex']!r} flag={o['sex_missing']}"
            )

        # Localization: unchanged up to strip/lowercase; flag reflects source
        # emptiness (a non-missing source 'unknown' stays 'unknown' with flag 0).
        expected_loc = s["localization"].strip().lower()
        expected_flag = "0" if s["localization"].strip() else "1"
        if o["localization"] != expected_loc or o["localization_missing"] != expected_flag:
            raise ValueError(
                f"Localization changed for image {image_id}: "
                f"source={s['localization']!r} out={o['localization']!r} "
                f"flag={o['localization_missing']}"
            )


# ------------------------------------------------------------------ statistics


def _print_statistics(rows, source_sex, source_loc):
    n = len(rows)
    unique_lesions = len({r["lesion_id"] for r in rows})
    missing_age = sum(r["age_missing"] for r in rows)
    missing_sex = sum(r["sex_missing"] for r in rows)
    missing_loc = sum(r["localization_missing"] for r in rows)
    sex_counts = Counter(r["sex"] for r in rows if not r["sex_missing"])
    loc_counts = Counter(r["localization"] for r in rows if not r["localization_missing"])
    class_counts = {c: sum(r[c] for r in rows) for c in CLASS_ORDER}

    print("\n" + "=" * 60)
    print(f"Ground truth schema: {SCHEMA_VERSION} ({EXPERIMENT_ID})")
    print("Metadata features: age, sex, localization (raw/categorical; no encoding)")
    print("Missing representation: empty string + *_missing=1 (never 'unknown')")
    print("Targets: 7-class one-hot")
    print("Identifiers: image, lesion_id")
    print("=" * 60)
    print(f"Total images: {n}")
    print(f"Total unique lesions: {unique_lesions}")
    print(f"\nSex:")
    for c in sorted(sex_counts):
        print(f"    {c} = {sex_counts[c]}")
    print(f"    missing = {missing_sex}")
    print(f"\nLocalization:")
    for c in sorted(loc_counts):
        print(f"    {c} = {loc_counts[c]}")
    print(f"    missing = {missing_loc}")
    print(f"\nAge missing: {missing_age}")
    print("\nOriginal source sex values (non-empty, as observed): "
          f"{', '.join(source_sex)}")
    print("Original source localization values (non-empty, as observed): "
          f"{', '.join(source_loc)}")
    print("\nClass counts (all 7 diagnoses):")
    for c in CLASS_ORDER:
        print(f"  {c}: {class_counts[c]}")
    print("=" * 60)


# ------------------------------------------------------------------ main


def main():
    if not os.path.exists(META):
        raise FileNotFoundError(f"Source metadata not found: {META}")

    with open(META, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        source_rows = list(reader)

    missing_cols = [c for c in REQUIRED_SOURCE_COLUMNS if c not in header]
    if missing_cols:
        raise ValueError(
            f"Source metadata is missing required column(s): {', '.join(missing_cols)}"
        )

    # Source-driven audit: the canonical sex set must match what the ORIGINAL
    # source actually contains — unexpected values need investigation, not
    # silent mutation.
    source_sex, source_loc = _collect_source_observations(source_rows)
    unexpected = set(source_sex) - SEX_CANONICAL
    if unexpected:
        raise ValueError(
            f"Unexpected sex value(s) in source metadata: {sorted(unexpected)}. "
            "Investigate before building the ground truth."
        )

    out_rows = []
    seen_images = set()
    for r in source_rows:
        image_id = r["image_id"].strip()
        if not image_id:
            raise ValueError("Empty image_id found in source metadata")
        if image_id in seen_images:
            raise ValueError(f"Duplicate image_id in source metadata: {image_id}")
        seen_images.add(image_id)

        lesion_id = r["lesion_id"].strip()
        if not lesion_id:
            raise ValueError(f"Empty lesion_id for image {image_id}")

        dx = r["dx"].strip().lower()
        if dx not in DX_TO_CLASS:
            raise ValueError(f"Unknown dx '{r['dx']}' for image {image_id}")
        label = DX_TO_CLASS[dx]

        age, age_missing = _parse_age(r["age"], image_id)
        sex, sex_missing = _normalize_sex(r["sex"], image_id)
        localization, localization_missing = _normalize_localization(r["localization"])

        row = {
            "image": image_id,
            "lesion_id": lesion_id,
            "age": "" if age is None else age,
            "age_missing": age_missing,
            "sex": sex,
            "sex_missing": sex_missing,
            "localization": localization,
            "localization_missing": localization_missing,
        }
        for c in CLASS_ORDER:
            row[c] = 1 if c == label else 0
        out_rows.append(row)

    # §12: strict structural + semantic validation before writing.
    _validate_strict(out_rows, source_rows)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    # §13: independent reload + row-by-row comparison against the original.
    _verify_against_source(OUT)

    _print_statistics(out_rows, source_sex, source_loc)
    print(f"\nWrote {len(out_rows)} rows to {OUT}")
    print("Target one-hot validation: PASS")
    print("Source-to-ground-truth semantic comparison: PASS")


if __name__ == "__main__":
    main()
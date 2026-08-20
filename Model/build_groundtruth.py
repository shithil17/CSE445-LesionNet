import csv

import os

META = "../Dataset/HAM10000_metadata.csv"
OUT = "../Dataset/HAM10000_groundtruth.csv"

CLASS_ORDER = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
DX_TO_CLASS = {c.lower(): c for c in CLASS_ORDER}


def main():
    with open(META, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    out_rows = []
    for r in rows:
        image_id = r["image_id"].strip()
        dx = r["dx"].strip().lower()
        if dx not in DX_TO_CLASS:
            raise ValueError(f"Unknown dx '{dx}' for {image_id}")
        out_rows.append(
            {
                "image": image_id,
                **{c: (1 if c == DX_TO_CLASS[dx] else 0) for c in CLASS_ORDER},
            }
        )

    columns = ["image"] + CLASS_ORDER
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {OUT}")


if __name__ == "__main__":
    main()

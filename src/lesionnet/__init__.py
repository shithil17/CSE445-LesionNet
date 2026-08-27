import argparse
from pathlib import Path

from lesionnet.app import launch
from lesionnet.config import CLASS_NAMES
from lesionnet.predict import predict_full

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


def run_cli(image_path, gender, age):
    result = predict_full(image_path, gender, age, output_dir=OUTPUT_DIR)
    print(f"Image: {image_path}")
    print(f"Top prediction: {result['pred_class']} ({result['pred_conf'] * 100:.2f}%)")
    for name, prob in zip(CLASS_NAMES, result["probs"]):
        print(f"  {name}: {prob * 100:.2f}%")
    print(f"Overlay saved: {result['overlay']}")
    print(f"Composite saved: {result['composite_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="lesionnet")
    parser.add_argument(
        "--cli",
        metavar="IMAGE",
        help="run backend headless on IMAGE and save outputs instead of launching the UI",
    )
    parser.add_argument("--gender", default="Prefer not to say")
    parser.add_argument("--age", type=int)
    args = parser.parse_args()

    if args.cli:
        run_cli(args.cli, args.gender, args.age)
    else:
        launch()


if __name__ == "__main__":
    main()
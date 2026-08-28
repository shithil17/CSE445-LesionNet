# LesionNet

Skin-lesion classifier (HAM10K, 7 classes) built on an already-trained
EfficientNetB4, wrapped in a Gradio web UI with Grad-CAM explainability.

The app loads the trained checkpoint as-is (**no retraining**) and serves both a
browser UI and a headless CLI.

## Usage

Run from the repo root.

### Web UI

```bash
uv run lesionnet
```

Opens the Gradio UI at http://0.0.0.0:7860. Upload a lesion image, optionally
pick gender/age (display only — the classifier is image-only), click **Submit**
to see all 7 class confidences plus a Grad-CAM overlay, then **Download result**
for a composite PNG with prediction, confidence bars, and metadata.

### Headless CLI

```bash
uv run lesionnet --cli <image> [--gender X --age N]
```

Runs the same backend without a server and writes `overlay.png` + `result.png`
to `<repo>/outputs/`.

```bash
uv run lesionnet --cli Model/HAM10000_split/train/MEL/ISIC_0024315.jpg --gender Female --age 63
```

## Model path

The checkpoint loads from `<repo>/Model/efficientnetb4_classifier.pth` by
default. Override with the `LESIONNET_MODEL_PATH` environment variable:

```bash
LESIONNET_MODEL_PATH=/path/to/classifier.pth uv run lesionnet
```

The architecture must match the training script exactly (EfficientNetB4 +
`Dropout(0.2)` + `Linear(1792, 7)`); the loader refuses to run on a state_dict
mismatch.

## Classes

`AKIEC, BCC, BKL, DF, MEL, NV, VASC` (ImageFolder alphabetical order).
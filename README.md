# Pokemon Card Rarity Classifier

CNN-based image classification of Pokemon Trading Card Game (TCG) cards into 5 rarity classes:
**Common · Uncommon · Rare · Ultra Rare · Secret Rare**.

---

## Setup

Requires **Python 3.10+**. Tested on macOS (Apple M1, 8 GB RAM) with `tensorflow-metal` for GPU
acceleration.

```bash
# Recommended: use uv (fast and reproducible)
uv sync

# Or: standard venv + pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Apple Silicon note**: `tensorflow-metal==1.2.0` is pinned in `pyproject.toml` and provides Metal
> GPU acceleration on M-series Macs. On other platforms you may need to adjust the TensorFlow
> dependency.

---

## How to run

Notebooks are numbered and meant to be run **in order** the first time:

| #   | Notebook                          | What it does                                              | Required to re-run? |
| --- | --------------------------------- | --------------------------------------------------------- | ------------------- |
| 1   | `1_data_exploration.ipynb`        | EDA: class distribution, image sizes, etc.                | No                  |
| 2   | `2_data_preparation.ipynb`        | Downloads cards, balances 500/class, builds `*.npy` files | Once (cached after) |
| 3   | `3_model_selection.ipynb`         | Random benchmark, simple ANN, basic CNN                   | No                  |
| 4   | `4_overfitting_experiment.ipynb`  | Baseline / Dropout / BatchNorm / L2 with n=3 multi-run    | No                  |
| 5   | `5_architecture_experiment.ipynb` | Architectural CNN variants                                | No                  |
| 6   | `6_transfer_learning.ipynb`       | MobileNetV2 transfer learning + fine-tuning               | No                  |
| 7   | `7_hyperparameter_search.ipynb`   | Grid search over LR × dropout for the chosen CNN          | No                  |
| 8   | `8_model_interpretation.ipynb`    | Grad-CAM heatmaps explaining what the CNN looks at        | No                  |

After the first run of notebook 2, the prepared image tensor is cached in
`data/dataset/x_images.npy` (~628 MB) and notebooks 3–8 load directly from that file.

---

## Project Structure

The overall structure of this project (notebooks, data, models, evaluation modules) follows the example project from [frank-trollmann/machine-learning_example-project](https://github.com/frank-trollmann/machine-learning_example-project.git).

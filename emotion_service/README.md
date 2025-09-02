# emotion_service — README

**Summary**

`emotion_service` is a modular pipeline for building, training and evaluating multimodal emotion models (audio + image / face) and for running anomaly detection on the resulting embeddings. It includes preprocessing utilities, embedding extraction, paired HDF5 builders, training scripts (multimodal classification), and an anomaly detector (Isolation Forest). The repo expects datasets formatted into image and audio folders (examples below: CREMA‑D and AffectNetAligned).

Datasets referenced in this project:
- CREMA‑D (audio + video): https://www.kaggle.com/datasets/ejlok1/cremad
- AffectNet Aligned (face images): https://www.kaggle.com/datasets/yakhyokhuja/affectnetaligned

---

## Goals / Capabilities

- Preprocess raw audio into fixed- or variable-length features and store them in HDF5 + CSV manifest.
- Extract image and audio embeddings (via pre-trained models or learned encoders) and save them as HDF5 files (`embeddings_images.h5`, `embeddings_audio.h5`).
- Build paired HDF5s that align sample filenames across modalities for multimodal training and evaluation.
- Train multimodal classifiers (image + audio) that can handle missing modalities and variable-length features.
- Train an Isolation Forest anomaly detector on the concatenated embeddings and visualize/save scores.

---

## Requirements / Installation

The project uses Python 3.8+. Install dependencies with:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` or want the core packages, a minimal list includes:

```
numpy pandas librosa soundfile h5py tqdm
torch torchvision torchaudio
scikit-learn joblib matplotlib pillow opencv-python

# optional: for web/frontend
flask fastapi uvicorn
```

Some scripts call `librosa`/`soundfile` for audio I/O and use `h5py` for HDF5. Training scripts expect PyTorch.

---

## Quick start — recommended pipeline

1. **Prepare datasets**
   - Place CREMA‑D audio files in a folder `data/audio/` (or change CLI flags accordingly).
   - Place AffectNet-aligned face images in `data/images/` and ensure filenames align between modalities if you want paired samples.

2. **Preprocess audio → HDF5 + manifest**

```bash
# from repo root (emotion_service)
python preprocessing_audio.py \
  --audio_dir data/audio \
  --out_h5 preprocessed_audio.h5 \
  --manifest_name audio_manifest.csv \
  --fixed_duration 3.0 \
  --workers 6
```

Key flags:
- `--audio_dir` : directory with audio files
- `--out_h5` : output h5 file
- `--manifest_name` : CSV manifest name (records filename, duration, label, etc.)
- `--fixed_duration` : if > 0, force features to fixed length (faster batching)
- `--workers` : number of processes for feature extraction

3. **Extract image & audio embeddings**

```bash
python extract_embeddings.py --images_dir data/images --out_h5 embeddings_images.h5 --model resnet18
python extract_embeddings.py --audio_h5 preprocessed_audio.h5 --out_h5 embeddings_audio.h5 --model audio_cnn
```

`extract_embeddings.py` should accept either a raw image folder or preprocessed audio H5 and export embeddings (typ. float32 arrays) to named datasets inside HDF5.

4. **Make paired HDF5 (align modalities)**

```bash
python make_paired_h5_filtered.py --img_h5 embeddings_images.h5 --aud_h5 embeddings_audio.h5 --out_h5 paired_embeddings.h5
```

The paired H5 contains datasets for image features, audio features, filenames and optional labels — shaped for training/evaluation.

5. **Train multimodal model**

```bash
python train_multimodal.py \
  --train_h5 paired_embeddings_train.h5 \
  --val_h5 paired_embeddings_val.h5 \
  --epochs 30 \
  --batch_size 64 \
  --lr 1e-3 \
  --save_dir saved_models
```

Important arguments (common):
- `--train_h5` / `--val_h5` : H5 files with paired embeddings
- `--batch_size` / `--epochs` / `--lr`
- `--workers` : DataLoader workers
- `--prefer_pairs_base` : option used in dataset helper (matching strategy)
- `--device` : CPU/GPU selection (auto-detects CUDA)

After training a `multimodal_best.pth` checkpoint will be saved to `saved_models/`.

6. **Train anomaly detector (Isolation Forest)**

```bash
python train_anomaly.py \
  --train_h5 paired_embeddings_train.h5 \
  --test_h5 paired_embeddings_test.h5 \
  --save_dir results/anomaly_iforest \
  --n_estimators 100
```

This script will concatenate image+audio embeddings (when available), fit `sklearn.ensemble.IsolationForest`, save the fitted `joblib` model and produce score arrays and optional plots (score histograms, PCA scatter) in `results/anomaly_iforest/`.

7. **Inference / prediction**

There is a lightweight inference script that accepts video/image/audio inputs and outputs emotion predictions (or anomaly scores). Example (as referenced in `train_multimodal.py` docstring):

```bash
python inference/predict.py --video test2.mp4 --multimodal saved_models/multimodal_best.pth --isolation results/anomaly/isolation_forest.joblib
```

Or use the API:

```bash
python app.py  # starts the demo API / UI (check the file for host/port flags)
```

---

## File-by-file summary (how to use)

- `preprocessing_audio.py` — converts raw WAV/MP3 to features stored inside an HDF5 and writes a manifest CSV. Supports fixed-duration framing and variable lengths. Use this as the first step for audio.

- `extract_embeddings.py` — extract embeddings from images or audio features. For images it typically runs a pre-trained CNN backbone (ResNet, etc.) and for audio it runs a pre-trained audio encoder or a simple CNN/MLP path.

- `make_paired_h5_filtered.py` — aligns image and audio embeddings by filename, optionally filters mismatches, and writes a paired HDF5 that training scripts consume.

- `train_multimodal.py` — PyTorch training loop that:
  - constructs dataset & streaming HDF5 loader
  - builds simple MLP encoders if needed for 1-D embeddings
  - supports missing modalities by zero-filling
  - checkpoints best model to `saved_models/`

- `train_anomaly.py` — trains an IsolationForest on concatenated embeddings. Saves `joblib` model and visualizations.

- `inspect_h5.py` / `inspect_model.py` — small utilities to inspect dataset shapes, example entries, and model architectures.

- `app.py` — demo server / API. Check the file to see available endpoints (e.g., `/predict`, `/upload`) and how to run.

- `frontend/` — minimal UI to demo predictions and visualizations. Run `app.py` and open the frontend URL in the browser (check the API base path in the code).

---

## Tips, gotchas & troubleshooting

- **HDF5 concurrency**: feature extraction uses `ProcessPoolExecutor` (or multiprocessing). HDF5 writes should be performed by the main process only to avoid corruption. Preprocessing scripts in this repo implement this pattern; if you get corrupted H5, re-run and reduce `--workers`.

- **Filename alignment**: paired datasets require consistent filenames (or a shared ID). Use the provided manifest CSVs to guarantee alignment.

- **Memory usage**: `train_multimodal.py` supports streaming HDF5 loading so you don't need to load the entire dataset into memory. Still, large batch sizes + worker count can increase RAM usage.

- **Variable-length audio**: either pad/trim to `--fixed_duration` for fixed-shape tensors, or store variable-length features and use a collate function during training.

- **Reproducibility**: seed the RNGs (PyTorch, NumPy, random) if you need reproducible results — training scripts include basic seeding lines.

---

## Example experiment reproducible steps (short)

1. Preprocess audio:
```bash
python preprocessing_audio.py --audio_dir data/cremad/audio --out_h5 preprocessed_cremad_audio.h5 --fixed_duration 3.0
```

2. Extract embeddings for Images & Audio:
```bash
python extract_embeddings.py --images_dir data/affectnet/aligned --out_h5 embeddings_images.h5
python extract_embeddings.py --audio_h5 preprocessed_cremad_audio.h5 --out_h5 embeddings_audio.h5
```

3. Build paired dataset:
```bash
python make_paired_h5_filtered.py --img_h5 embeddings_images.h5 --aud_h5 embeddings_audio.h5 --out_h5 paired_cremad_affectnet.h5
```

4. Train multimodal model:
```bash
python train_multimodal.py --train_h5 paired_cremad_affectnet_train.h5 --val_h5 paired_cremad_affectnet_val.h5 --epochs 25
```

5. Optionally train/read anomaly detector:
```bash
python train_anomaly.py --train_h5 paired_cremad_affectnet_train.h5 --test_h5 paired_cremad_affectnet_test.h5 --save_dir results/anomaly
```

---

## Where outputs are stored

- `*.h5` — preprocessed features & embeddings
- `saved_models/` — model checkpoints (best / last)
- `results/` — evaluation metrics, plots, anomaly scores, joblib models

---

## Contact / author

If you want edits to this documentation (add specific command flags, include exact function signatures or CLI help text), tell me which script you want me to read in full and I will extract the exact flags & defaults from the code and update the README accordingly.

---


## Exact CLI flags & defaults (extracted from scripts)

### `preprocessing_audio.py`

| Option(s) | Keyword args (type/default/help/action/dest) | Source snippet |
|---|---|---|
| `--audio_dir` | `required` = `True`<br>`help` = `Root folder containing audio files` | `parser.add_argument('--audio_dir', required=True, help='Root folder containing audio files')` |
| `--out_h5` | `required` = `False`<br>`default` = `None`<br>`help` = `Output HDF5 file path (default: ./preprocessed_audio.h5)` | `parser.add_argument('--out_h5', default=None, help='Output HDF5 file path (default: ./preprocessed_audio.h5)')` |
| `--out_dir` | `required` = `False`<br>`default` = `None`<br>`help` = `If set, write manifest CSV to this folder (defaults to project preprocessed_audio)` | `parser.add_argument('--out_dir', default=None, help='If set, write manifest CSV to this folder (defaults to project preprocessed_audio)')` |
| `--sr` | `type` = `int`<br>`default` = `16000`<br>`help` = `Target sample rate` | `parser.add_argument('--sr', type=int, default=16000, help='Target sample rate')` |
| `--n_fft` | `type` = `int`<br>`default` = `1024` | `parser.add_argument('--n_fft', type=int, default=1024)` |
| `--hop_length` | `type` = `int`<br>`default` = `512` | `parser.add_argument('--hop_length', type=int, default=512)` |
| `--n_mels` | `type` = `int`<br>`default` = `64` | `parser.add_argument('--n_mels', type=int, default=64)` |
| `--fixed_duration` | `type` = `float`<br>`default` = `3.0`<br>`help` = `If >0, force fixed duration (seconds)` | `parser.add_argument('--fixed_duration', type=float, default=3.0, help='If >0, force fixed duration (seconds)')` |
| `--workers` | `type` = `int`<br>`default` = `4` | `parser.add_argument('--workers', type=int, default=4)` |
| `--manifest_name` | `default` = `audio_manifest.csv` | `parser.add_argument('--manifest_name', default='audio_manifest.csv')` |


### `train_multimodal.py`

| Option(s) | Keyword args (type/default/help/action/dest) | Source snippet |
|---|---|---|
| `--train_h5` | `required` = `True` | `parser.add_argument('--train_h5', required=True, help='Training paired HDF5')` |
| `--val_h5` | `required` = `False`<br>`default` = `None` | `parser.add_argument('--val_h5', default=None, help='Validation paired HDF5')` |
| `--epochs` | `type` = `int`<br>`default` = `30` | `parser.add_argument('--epochs', type=int, default=30)` |
| `--batch_size` | `type` = `int`<br>`default` = `64` | `parser.add_argument('--batch_size', type=int, default=64)` |
| `--lr` | `type` = `float`<br>`default` = `0.001` | `parser.add_argument('--lr', type=float, default=1e-3)` |
| `--device` | `default` = `auto` | `parser.add_argument('--device', default='auto')` |
| `--save_dir` | `default` = `saved_models` | `parser.add_argument('--save_dir', default='saved_models')` |
| `--workers` | `type` = `int`<br>`default` = `4` | `parser.add_argument('--workers', type=int, default=4)` |
| `--seed` | `type` = `int`<br>`default` = `42` | `parser.add_argument('--seed', type=int, default=42)` |
| `--prefetch` | `action` = `store_true` | `parser.add_argument('--prefetch', action='store_true')` |


### `train_anomaly.py`

| Option(s) | Keyword args (type/default/help/action/dest) | Source snippet |
|---|---|---|
| `--train_h5` | `required` = `True` | `parser.add_argument('--train_h5', required=True, help='Training HDF5 for anomaly detector')` |
| `--test_h5` | `default` = `None` | `parser.add_argument('--test_h5', default=None)` |
| `--save_dir` | `default` = `saved_models` | `parser.add_argument('--save_dir', default='saved_models')` |
| `--n_estimators` | `type` = `int`<br>`default` = `100` | `parser.add_argument('--n_estimators', type=int, default=100)` |
| `--contamination` | `type` = `float`<br>`default` = `0.01` | `parser.add_argument('--contamination', type=float, default=0.01)` |
| `--random_state` | `type` = `int`<br>`default` = `42` | `parser.add_argument('--random_state', type=int, default=42)` |
| `--eval_with_labels` | `action` = `store_true`<br>`help` = `If set, try a simple binary eval when labels look binary (0/1)` | `parser.add_argument('--eval_with_labels', action='store_true', help='If set, try a simple binary eval when labels look binary (0/1)')` |

---


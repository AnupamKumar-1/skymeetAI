# SkymeetAI - Emotion Service

## Project Overview

This project implements a multimodal system for emotion recognition and anomaly detection using audio and visual (facial) inputs. It combines deep learning models for feature extraction and fusion, trained on paired audio-image data, with an Isolation Forest for detecting anomalous inputs (e.g., out-of-distribution emotions or data artifacts).

Key features:
- **Preprocessing**: Handles audio (mel spectrograms) and images (face cropping, resizing).
- **Embedding Extraction**: Uses ResNet18 for images and a simple MLP for audio features.
- **Pairing**: Aligns audio and image embeddings by emotion labels for training.
- **Multimodal Training**: Fuses embeddings into a classifier for emotions (e.g., angry, happy, sad, neutral).
- **Anomaly Detection**: Trains an Isolation Forest on fused embeddings to flag anomalies.
- **Inference**: Predicts emotions and anomaly scores from audio/video inputs.
- **API**: A Flask-based endpoint for real-time analysis.
- **Evaluation**: Metrics, confusion matrices, and plots for model performance.

The system is designed for applications like meeting analysis, sentiment monitoring, or affective computing.

## Datasets

This project uses the following publicly available datasets:

- **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**:
  - Source: [Kaggle - CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)
  - Description: Audio recordings of actors expressing emotions (anger, disgust, fear, happy, neutral, sad). Includes ~7,442 clips with emotion labels encoded in filenames.
  - Usage: Audio preprocessing and emotion pairing.
  - License: CC BY 4.0 (check Kaggle for details).
  - Preparation: Place audio files in a directory like `data/audio/`. Labels are extracted from filenames (e.g., 'ANG' → 'anger').

- **AffectNet Aligned (Facial Emotion Dataset)**:
  - Source: [Kaggle - AffectNet Aligned](https://www.kaggle.com/datasets/yakhyokhuja/affectnetaligned)
  - Description: ~420,000 aligned facial images labeled with 8 emotions (angry, calm, disgust, fear, happy, neutral, sad, surprise). Split into train/val/test.
  - Usage: Image preprocessing, face cropping, and emotion pairing.
  - License: For research/non-commercial use (refer to original AffectNet paper and Kaggle terms).
  - Preparation: Organize into folders like `data/images/train/{class_id}/image.jpg`. Class IDs map to emotions (e.g., 0=angry).

**Note**: Download datasets from the links above. Ensure ethical use: these datasets involve human subjects, so respect privacy and avoid biased applications. The project filters to common emotions (angry, disgust, fear, happy, neutral, sad) for pairing.

## Installation and Setup

### Requirements
- Python 3.8+
- Libraries: Install via `pip install -r requirements.txt`. The `requirements.txt` file specifies the following dependencies with version constraints for compatibility and reproducibility:

```
torch
numpy>=1.26.0,<2.0.0
pandas>=2.3.0
scikit-learn>=1.4.0
scipy>=1.11.0

librosa==0.11.0
audioread==3.0.1
soundfile>=0.12.1
soxr>=0.5.0.post1

Pillow>=10.0.0
mtcnn==1.0.0
opencv-python-headless<4.10.0

Flask>=3.0.0
click>=8.0.0
blinker>=1.6.0
itsdangerous>=2.1.0
Jinja2>=3.1.0

matplotlib>=3.8.0
tqdm>=4.65.0
joblib>=1.3.0

platformdirs>=4.0.0
pooch>=1.8.0
lz4>=4.0.0
gunicorn
torchvision
torchaudio
h5py
```

**Notes on Dependencies**:
- **Core ML**: `torch`, `torchvision`, `torchaudio` for PyTorch-based models and audio processing.
- **Data Handling**: `numpy`, `pandas`, `scipy`, `h5py` for arrays, dataframes, and HDF5 storage.
- **Audio**: `librosa`, `soundfile`, `audioread`, `soxr` for feature extraction (e.g., mel spectrograms).
- **Images**: `Pillow`, `opencv-python-headless`, `mtcnn` for processing, face detection, and alignment.
- **Web/API**: `Flask` and its dependencies (`click`, `blinker`, etc.) for the API server.
- **Viz & Utils**: `matplotlib` for plots, `tqdm` for progress, `joblib` for model serialization.
- **Deployment**: `gunicorn` for production serving, `lz4` for compression, `pooch`/`platformdirs` for caching/data management.

Install with: `pip install -r requirements.txt`. Some packages (e.g., `opencv-python-headless`) are optional but recommended for headless environments like servers.

### Environment Setup
1. Clone the repository: `git clone <repo-url> && cd emotion_service`
2. Install dependencies: `pip install -r requirements.txt`
3. Download datasets and place in `data/` folder.
4. (Optional) Set environment variables:
   - `FLASK_CORS_ORIGINS=http://localhost:3000` for API CORS.
   - `BACKEND_URL=http://your-backend/api` for forwarding results.
   - `LOG_LEVEL=DEBUG` for verbose logging.
   - Install system dependencies: `ffmpeg` for video processing (e.g., `apt-get install ffmpeg` on Ubuntu).

## Preprocessing

### Images
- Script: `preprocess_images.py`
- Usage: `python preprocess_images.py --src data/images --out preprocessed_images.h5 --size 224 --face_crop 1 --workers 4`
- Steps:
  - Scans folder structure (e.g., `train/{class_id}/img.jpg`).
  - Optional: Face cropping using OpenCV Haar cascade or MTCNN.
  - Center-crop to square, resize to 224x224, normalize to [0,1].
  - Saves to HDF5: Groups for train/val/test with datasets `images` (N,H,W,C), `labels` (int), `paths` (str).

### Audio
- Script: `preprocessing_audio.py`
- Usage: `python preprocessing_audio.py --audio_dir data/audio --out_h5 preprocessed_audio.h5 --fixed_duration 3.0 --workers 6`
- Steps:
  - Loads WAV files, resamples to 16kHz.
  - Trims silence, normalizes, pads/truncates to fixed duration (e.g., 3s).
  - Computes log-mel spectrograms (64 bands).
  - Saves to HDF5: `features/fixed` (N,64,frames), `labels`, `paths`. Also generates a CSV manifest.

### Embeddings Extraction
- Script: `extract_embeddings.py`
- Usage: Run directly: `python extract_embeddings.py`
- Steps:
  - Loads preprocessed HDF5s.
  - Images: ResNet18 (pretrained) → 512-d embeddings.
  - Audio: MLP on flattened mel → 128-d embeddings.
  - Saves: `embeddings_images.h5` (split/embeddings (N,512)), `embeddings_audio.h5` (all/embeddings (N,128)).

### Pairing Embeddings
- Script: `make_paired_h5_filtered.py`
- Usage: `python make_paired_h5_filtered.py`
- Steps:
  - Pairs image-audio embeddings by common emotions (angry, disgust, fear, happy, neutral, sad).
  - Balances classes via oversampling audio pools.
  - Saves per-split: `saved_paired_{split}_paired_embeddings.h5` with `image_embeddings` (N,512), `audio_embeddings` (N,128), `labels`, `paths`.

## Training

### Multimodal Emotion Classifier
- Script: `train_multimodal.py`
- Usage: `python train_multimodal.py --train_h5 saved_paired_train_paired_embeddings.h5 --val_h5 saved_paired_val_paired_embeddings.h5 --test_h5 saved_paired_test_paired_embeddings.h5 --epochs 50 --batch_size 128 --lr 0.001 --save_dir saved_models`
- Architecture:
  - Image: ResNet18 (pretrained) + MLP → 256-d.
  - Audio: MLP on 128-d → 256-d.
  - Fusion: Concat → Linear → Softmax (num_classes=6).
- Features: Class weighting, OneCycleLR scheduler, validation.
- Output: Saves best checkpoint `multimodal_best.pth` with state dict and class labels.

### Anomaly Detector (Isolation Forest)
- Script: `train_anomaly.py`
- Usage: `python train_anomaly.py --train_h5 saved_paired_train_paired_embeddings.h5 --test_h5 saved_paired_test_paired_embeddings.h5 --save_dir results/anomaly --n_estimators 200 --contamination 0.01`
- Steps:
  - Concatenates image+audio embeddings (N,640).
  - Fits IsolationForest.
- Output: `isolation_forest.joblib`, anomaly scores (.npz), plots (histograms, PCA scatters).

## Evaluation and Plotting

- Script: `train_eval_plot.py`
- Usage: `python train_eval_plot.py --model_path saved_models/multimodal_best.pth --test_h5 saved_paired_test_paired_embeddings.h5 --val_h5 saved_paired_val_paired_embeddings.h5 --save_dir results/plots`
- Computes: Accuracy, F1, classification report, confusion matrix.
- Plots: Confusion matrices (raw/normalized), test vs val accuracy, per-class F1.
- Saves: Raw predictions (.npz) for further analysis.

## Inference

### Command-Line Prediction
- Script: `predict.py`
- Usage:
  - Video: `python predict.py --video test.mp4 --multimodal saved_models/multimodal_best.pth --isolation results/anomaly/isolation_forest.joblib`
  - Image+Audio: `python predict.py --image frame.jpg --audio clip.wav --multimodal ... --isolation ...`
- Output: JSON with `emotion_probs` (dict of probabilities), `anomaly_score` (higher=more anomalous), `anomaly_flag` ("anomaly" or "normal").

### API Endpoint
- Script: `app.py`
- Usage: Run `python app.py` (listens on port 5002). For production: `gunicorn -w 4 app:app`.
- Endpoint: POST `/analyze`
  - Form-data: `meeting_id` (str), `participant_id` (str), `file` (audio/video file), `type` (audio|video).
- Response: JSON with timeline, anomalies, emotions.
- Features: Preloads models, CORS support, optional backend forwarding.
- Development: `FLASK_ENV=development python app.py`

## Architecture Diagrams

### Overall System Architecture
The system follows a pipeline from data ingestion to inference. Below is a textual representation (ASCII art):

```
+-------------+     +-------------+
|   CREMA-D   |     | AffectNet   |
|   (Audio)   |     |  (Images)   |
+------+------+     +------+------+
       |                   |
       v                   v
+------+------+     +------+------+
| Preprocess  |     | Preprocess  |
| Audio       |     | Images      |
+------+------+     +------+------+
       |                   |
       v                   v
+------+------+     +------+------+
| Extract     |     | Extract     |
| Embeddings  |     | Embeddings  |
| (MLP)       |     | (ResNet18)  |
+------+------+     +------+------+
       |                   |
       +---------+---------+
                 |
                 v
        +--------+--------+
        | Pair Embeddings |
        | by Emotion     |
        +--------+--------+
                 |
                 v
        +--------+--------+
        | Train Multimodal|
        | Fusion Model    |
        +--------+--------+
                 |
                 v
        +--------+--------+
        | Train Anomaly   |
        | Detector (IF)   |
        +--------+--------+
                 |
                 v
        +--------+--------+
        | Inference:      |
        | Predict Emotions|
        | & Anomalies     |
        +--------+--------+
```

### Multimodal Model Architecture
Detailed structure of the emotion classifier (ASCII art):

```
Image Input (3x224x224)
      |
      v
ResNet18 (Pretrained) --> 512-d Feature
      |
      v
MLP (Linear + ReLU) --> 256-d Embedding

Audio Input (Mel Spectrogram --> Flattened 128-d)
      |
      v
MLP (Linear + ReLU) --> 256-d Embedding

Concat (512-d Total)
      |
      v
Fusion Linear --> Logits (6 Classes: angry, disgust, fear, happy, neutral, sad)
      |
      v
Softmax --> Probabilities
```

### Anomaly Detection Flow
```
Paired Embeddings (Image 512-d + Audio 128-d) --> Concat (640-d)
      |
      v
Isolation Forest Fit (on Train Data)
      |
      v
Inference: Decision Function --> Anomaly Score
           Predict --> -1 (Anomaly) / 1 (Normal)
```


- Contributions: Welcome! Focus on improving fairness, reducing bias in emotion detection.
- License: MIT (assumed; adjust as needed).
- Citation: If using, cite the datasets and this project.
- Warnings: Emotion AI can be biased (e.g., cultural differences). Use responsibly for research only.

For issues, open a GitHub ticket. Last updated: September 03, 2025.
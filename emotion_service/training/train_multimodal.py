
"""
train_multimodal.py

Streaming HDF5 loader version — does NOT load full arrays into memory.

This version is more robust for paired embedding HDF5s (shape (N, D))
and will automatically build MLP encoders for 1-D embeddings.
Saves best model into emotion_service/saved_models by default.

emotion_service % python inference/predict.py --video test2.mp4 \
  --multimodal saved_models/multimodal_best.pth \
  --isolation results/anomaly/isolation_forest.joblib
"""
import argparse
import random
import re
import sys
import time
import math
import os
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models

# -------------------------
# Utilities for H5 loading
# -------------------------
def decode_if_bytes(x):
    if isinstance(x, bytes):
        try:
            return x.decode('utf-8')
        except:
            return x
    if isinstance(x, np.ndarray) and x.dtype.kind == 'S':
        # byte strings array
        return np.array([b.decode('utf-8') for b in x])
    return x


def list_all_datasets(h5):
    datasets = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            full = '/' + name
            datasets.append((full, obj))
    h5.visititems(visitor)
    return datasets

# -------------------------
# Label extraction helpers
# -------------------------
TOKEN_TO_LABEL = {
    'ang': 'angry', 'angry': 'angry', 'anger': 'angry',
    'hap': 'happy', 'happy': 'happy', 'happiness': 'happy',
    'sad': 'sad', 'sadness': 'sad',
    'neu': 'neutral', 'neutral': 'neutral',
    'dis': 'disgust', 'disgust': 'disgust',
    'fea': 'fear', 'fear': 'fear', 'fright': 'fear',
    'sur': 'surprise', 'surprise': 'surprise'
}


def extract_label_from_string(s: str) -> str:
    if s is None:
        return None
    s = str(s).lower()
    for token, label in TOKEN_TO_LABEL.items():
        if re.search(r'\b' + re.escape(token) + r'\b', s):
            return label
    for token, label in TOKEN_TO_LABEL.items():
        if token in s:
            return label
    return None

# -------------------------
# Robust HDF5 scanner
# -------------------------
def scan_h5_for_dataset(h5_path: str, kind: str):
    """
    Open HDF5 read-only briefly to find the most-likely data dataset path,
    plus paths for filenames and labels (if present).
    Returns {'data_path','data_shape','data_dtype','filenames','labels'}.
    """
    print(f"[scanner] scanning {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        datasets = list_all_datasets(f)
        if not datasets:
            raise RuntimeError(f"No datasets found in {h5_path}")

        info = []
        for path, ds in datasets:
            try:
                shape = ds.shape
                dtype = ds.dtype
                # guard: some very large products overflow int on weird platforms, cast carefully
                try:
                    prod = int(np.prod(shape))
                except Exception:
                    prod = 0
                info.append((path, shape, dtype, prod))
            except Exception:
                continue

        # If every dataset has shape length 0 (scalars) or something unexpected,
        # we'll still try to pick a dataset by name or by largest first-dim.
        data_candidates = [(p, s, d, prod) for (p, s, d, prod) in info if len(s) >= 1]

        # prefer by name hints depending on kind
        if kind == 'image':
            hints = ['image', 'img', 'face', 'visual', 'image_embedding', 'image_emb', 'emb', 'embedding']
            # prefer 2-D or higher (embeddings or images) then 3/4D images
            candidates = [c for c in data_candidates if len(c[1]) >= 2]
            if not candidates:
                candidates = data_candidates[:]
        else:
            # audio / embedding hints
            hints = ['audio', 'aud', 'speech', 'wav', 'mel', 'mfcc', 'emb', 'embedding', 'embeddings', 'feat', 'feature']
            candidates = [c for c in data_candidates if len(c[1]) >= 2]
            if not candidates:
                candidates = data_candidates[:]

        # try to pick by name hints first
        picked = None
        for path, shape, dtype, prod in candidates:
            low = path.lower()
            for tok in hints:
                if tok in low:
                    picked = (path, shape, dtype, prod)
                    break
            if picked:
                break

        # if not found by name, pick the dataset with the largest first-dimension N (shape[0]),
        # falling back to largest product if needed.
        if picked is None and candidates:
            # prefer largest shape[0]
            best = None
            best_first = -1
            for path, shape, dtype, prod in candidates:
                first = int(shape[0]) if len(shape) >= 1 else 0
                if first > best_first:
                    best_first = first
                    best = (path, shape, dtype, prod)
            if best is None:
                # fallback to max product
                candidates.sort(key=lambda x: x[3], reverse=True)
                best = candidates[0]
            picked = best

        if picked is None:
            # As a last resort, try to pick any dataset
            if info:
                picked = info[0]
            else:
                raise RuntimeError(f"Could not determine main dataset in {h5_path}")

        data_path, data_shape, data_dtype, _ = picked

        # now try to find 1-D arrays of length N for filenames/labels
        N = int(data_shape[0]) if len(data_shape) >= 1 else None
        fname_cands = []
        label_cands = []
        for path, shape, dtype, _ in info:
            if N is None:
                continue
            if len(shape) == 1 and shape[0] == N:
                fname_cands.append((path, dtype))
                label_cands.append((path, dtype))
            if len(shape) == 2 and shape[0] == N and shape[1] in (1,):
                fname_cands.append((path, dtype))
                label_cands.append((path, dtype))

        def pick_named(cands, hints2):
            for path, _dtype in cands:
                low = path.lower()
                for tok in hints2:
                    if tok in low:
                        return path
            return None

        fname_path = pick_named(fname_cands, ['filename', 'filenames', 'name', 'file', 'audio_names', 'image_names'])
        label_path = pick_named(label_cands, ['label', 'labels', 'y', 'emotion', 'emotion_labels'])
        if fname_path is None and fname_cands:
            fname_path = fname_cands[0][0]
        if label_path is None and label_cands:
            label_path = label_cands[0][0]

        filenames = None
        labels = None
        if fname_path is not None:
            try:
                raw = np.array(f[fname_path])
                raw = decode_if_bytes(raw)
                filenames = [str(x) for x in raw]
            except Exception:
                filenames = None
        if label_path is not None:
            try:
                raw = np.array(f[label_path])
                raw = decode_if_bytes(raw)
                labels = [str(x) for x in raw]
            except Exception:
                labels = None

        meta = {
            "data_path": data_path,
            "data_shape": tuple(data_shape),
            "data_dtype": data_dtype,
            "filenames": filenames,
            "labels": labels
        }
        print(f"[scanner] found data {meta['data_path']} shape={meta['data_shape']} dtype={meta['data_dtype']}")
        return meta


# New helper: scan a single H5 file for a paired (image/audio) embedding pair
def scan_paired_h5(h5_path: str):
    """
    When both image and audio embeddings are stored in the same H5 file, try to locate
    the two best embedding datasets and their optional filename/label arrays.
    Returns (img_meta, aud_meta) compatible with scan_h5_for_dataset output.
    """
    print(f"[paired-scanner] scanning paired file {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        datasets = list_all_datasets(f)
        info = []
        for path, ds in datasets:
            try:
                shape = ds.shape
                dtype = ds.dtype
                try:
                    prod = int(np.prod(shape))
                except Exception:
                    prod = 0
                info.append((path, shape, dtype, prod))
            except Exception:
                continue

        # consider only candidates with at least 2D (N,D) or 1D but reasonable
        emb_candidates = [(p, s, d, prod) for (p, s, d, prod) in info if len(s) >= 2]
        if not emb_candidates:
            emb_candidates = info[:]

        # scoring: prefer names with obvious tokens
        img_hints = ['image', 'img', 'visual', 'face', 'image_embedding', 'image_emb']
        aud_hints = ['audio', 'aud', 'speech', 'wav', 'mel', 'mfcc', 'audio_embedding', 'audio_emb']

        def score_name(path, hints):
            low = path.lower()
            for i, tok in enumerate(hints):
                if tok in low:
                    return len(hints) - i
            return 0

        scored = []
        for path, shape, dtype, prod in emb_candidates:
            s = max(score_name(path, img_hints), score_name(path, aud_hints))
            scored.append((path, shape, dtype, prod, s))

        # sort by score then by first-dim size
        scored.sort(key=lambda x: (x[4], int(x[1][0]) if len(x[1])>=1 else 0, x[3]), reverse=True)

        if len(scored) >= 2:
            # try to pick two with different sizes or names
            img_choice = scored[0]
            aud_choice = None
            for cand in scored[1:]:
                # prefer different paths
                if cand[0] != img_choice[0]:
                    aud_choice = cand
                    break
            if aud_choice is None:
                aud_choice = scored[1]
        elif len(scored) == 1:
            img_choice = scored[0]
            aud_choice = None
        else:
            raise RuntimeError(f"No embedding-like datasets found in paired H5: {h5_path}")

        # heuristics: if the best two have very different dims (e.g., 512 vs 128) try to
        # decide which is image (likely larger) and which is audio (likely smaller)
        def pick_roles(a, b):
            pa, sa, da, proda, sa_score = a
            pb, sb, db, prodb, sb_score = b
            # prefer larger second-dimension as image embedding
            da_dim = sa[1] if len(sa) >= 2 else 1
            db_dim = sb[1] if len(sb) >= 2 else 1
            # if dims differ significantly use that
            if da_dim >= db_dim:
                return a, b
            else:
                return b, a

        if aud_choice is not None:
            img_sel, aud_sel = pick_roles(img_choice, aud_choice)
            img_meta = {"data_path": img_sel[0], "data_shape": tuple(img_sel[1]), "data_dtype": img_sel[2], "filenames": None, "labels": None}
            aud_meta = {"data_path": aud_sel[0], "data_shape": tuple(aud_sel[1]), "data_dtype": aud_sel[2], "filenames": None, "labels": None}
        else:
            img_sel = img_choice
            img_meta = {"data_path": img_sel[0], "data_shape": tuple(img_sel[1]), "data_dtype": img_sel[2], "filenames": None, "labels": None}
            aud_meta = {"data_path": img_sel[0], "data_shape": tuple(img_sel[1]), "data_dtype": img_sel[2], "filenames": None, "labels": None}

        # attempt to find 1-D filename/label arrays matching each N
        for path, shape, dtype, prod in info:
            if len(shape) == 1:
                # try assign to either if length matches
                try:
                    if shape[0] == img_meta['data_shape'][0] and img_meta.get('filenames') is None:
                        raw = np.array(f[path])
                        raw = decode_if_bytes(raw)
                        img_meta['filenames'] = [str(x) for x in raw]
                    if shape[0] == aud_meta['data_shape'][0] and aud_meta.get('filenames') is None:
                        raw = np.array(f[path])
                        raw = decode_if_bytes(raw)
                        aud_meta['filenames'] = [str(x) for x in raw]
                except Exception:
                    pass
            if len(shape) == 2 and shape[1] in (1,):
                try:
                    if shape[0] == img_meta['data_shape'][0] and img_meta.get('labels') is None:
                        raw = np.array(f[path])
                        raw = decode_if_bytes(raw)
                        img_meta['labels'] = [str(x) for x in raw]
                    if shape[0] == aud_meta['data_shape'][0] and aud_meta.get('labels') is None:
                        raw = np.array(f[path])
                        raw = decode_if_bytes(raw)
                        aud_meta['labels'] = [str(x) for x in raw]
                except Exception:
                    pass

        print(f"[paired-scanner] chosen image dataset {img_meta['data_path']} shape={img_meta['data_shape']}")
        print(f"[paired-scanner] chosen audio dataset {aud_meta['data_path']} shape={aud_meta['data_shape']}")
        return img_meta, aud_meta

# -------------------------
# Streaming paired dataset
# -------------------------
class EmotionPairDataset(Dataset):
    def __init__(self,
                 images_h5: str,
                 audios_h5: str,
                 image_transform=None,
                 audio_transform=None,
                 prefer_pairs_base='images'):
        super().__init__()
        self.images_h5_path = images_h5
        self.audios_h5_path = audios_h5
        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.prefer_pairs_base = prefer_pairs_base

        # scan files
        if os.path.abspath(images_h5) == os.path.abspath(audios_h5):
            img_meta, aud_meta = scan_paired_h5(images_h5)
        else:
            img_meta = scan_h5_for_dataset(images_h5, 'image')
            aud_meta = scan_h5_for_dataset(audios_h5, 'audio')

        self.image_data_path = img_meta['data_path']
        self.image_shape = img_meta['data_shape']
        self.image_dtype = img_meta['data_dtype']

        self.audio_data_path = aud_meta['data_path']
        self.audio_shape = aud_meta['data_shape']
        self.audio_dtype = aud_meta['data_dtype']

        self.image_fnames = img_meta.get('filenames', None)
        self.audio_fnames = aud_meta.get('filenames', None)
        self.image_labels_raw = img_meta.get('labels', None)
        self.audio_labels_raw = aud_meta.get('labels', None)

        self._images_h5 = None
        self._audios_h5 = None

        self.num_images = int(self.image_shape[0])
        self.num_audios = int(self.audio_shape[0])

        # audio features presence detection
        self.audio_features_present = True
        self.audio_feature_shape = tuple(self.audio_shape[1:]) if len(self.audio_shape) > 1 else (1,)
        try:
            with h5py.File(self.audios_h5_path, 'r') as _fhandle:
                ds = _fhandle[self.audio_data_path]
                if (ds.dtype.kind in ('S', 'U')) or (len(ds.shape) == 1 and not np.issubdtype(ds.dtype, np.number)):
                    print(f"[scanner-warning] audio dataset {self.audio_data_path} looks like labels/strings (shape={ds.shape}, dtype={ds.dtype}). Will use labels for pairing and use zero audio features.")
                    self.audio_features_present = False
                    self.audio_feature_shape = (1,)
        except Exception:
            pass

        # infer labels
        self.image_labels = self._infer_labels(self.image_fnames, self.image_labels_raw, self.num_images)
        self.audio_labels = self._infer_labels(self.audio_fnames, self.audio_labels_raw, self.num_audios)

        # numeric->emotion label map (optional)
        label_map = {
            '0': 'happy', '1': 'sad', '2': 'angry', '3': 'neutral',
            '4': 'fear', '5': 'disgust', '6': 'surprise', '7': 'calm'
        }
        try:
            unique_img_labels_before = sorted(set([str(x) for x in self.image_labels]))
            print(f"[label_map] unique image labels before mapping: {unique_img_labels_before}")
            if any(str(l) in label_map for l in self.image_labels):
                self.image_labels = [label_map.get(str(l), str(l)) for l in self.image_labels]
                unique_img_labels_after = sorted(set(self.image_labels))
                print(f"[label_map] applied mapping -> unique image labels after mapping: {unique_img_labels_after}")
            else:
                print("[label_map] no numeric image labels found that match mapping keys; skipping mapping.")
        except Exception as e:
            print(f"[label_map] warning: failed to apply label mapping: {e}")

        # build indices per label
        self.label2imgidxs = defaultdict(list)
        for i, lab in enumerate(self.image_labels):
            self.label2imgidxs[lab].append(i)

        self.label2audidxs = defaultdict(list)
        for i, lab in enumerate(self.audio_labels):
            self.label2audidxs[lab].append(i)

        image_label_set = set(self.label2imgidxs.keys())
        audio_label_set = set(self.label2audidxs.keys())
        self.common_labels = sorted(list(image_label_set.intersection(audio_label_set)))
        if not self.common_labels:
            print("WARNING: No overlapping emotion labels between images and audios. using image-only mode.")
            self.class_labels = sorted(list(image_label_set))
            self.label2idx = {l: i for i, l in enumerate(self.class_labels)}
            self.pairs = [(i, None) for i in range(self.num_images)]
            self.mode = 'image-only'
        else:
            self.class_labels = self.common_labels
            self.label2idx = {l: i for i, l in enumerate(self.class_labels)}
            self.mode = 'paired'
            self.resample_pairs()

    def _open_files_if_needed(self):
        if self._images_h5 is None:
            self._images_h5 = h5py.File(self.images_h5_path, 'r')
        if self._audios_h5 is None:
            # both may point to same file but that's OK to open twice
            self._audios_h5 = h5py.File(self.audios_h5_path, 'r')

    def close_files(self):
        if getattr(self, "_images_h5", None) is not None:
            try:
                self._images_h5.close()
            except Exception:
                pass
            self._images_h5 = None
        if getattr(self, "_audios_h5", None) is not None:
            try:
                self._audios_h5.close()
            except Exception:
                pass
            self._audios_h5 = None

    def __del__(self):
        try:
            self.close_files()
        except Exception:
            pass

    def _infer_labels(self, filenames, label_array, n_samples):
        labels_out = []
        if label_array is not None and len(label_array) == n_samples:
            for v in label_array:
                lab = extract_label_from_string(v)
                labels_out.append(lab if lab is not None else str(v))
            return labels_out
        if label_array is not None and len(label_array) != n_samples:
            print(f"[warning] label array length {len(label_array)} != expected {n_samples}; using what is available.")
            for v in label_array:
                lab = extract_label_from_string(v)
                labels_out.append(lab if lab is not None else str(v))
            if len(labels_out) < n_samples:
                labels_out += ['unknown'] * (n_samples - len(labels_out))
            else:
                labels_out = labels_out[:n_samples]
            return labels_out
        if filenames is not None and len(filenames) == n_samples:
            for fn in filenames:
                lab = extract_label_from_string(fn)
                labels_out.append(lab if lab is not None else 'unknown')
            return labels_out
        return ['unknown'] * n_samples

    def resample_pairs(self, num_pairs: int = None):
        if self.mode == 'image-only':
            self.pairs = [(i, None) for i in range(self.num_images)]
            return
        if num_pairs is None:
            num_pairs = self.num_images if self.prefer_pairs_base == 'images' else self.num_audios
        pairs = []
        if self.prefer_pairs_base == 'images':
            for i in range(self.num_images):
                lab = self.image_labels[i]
                aud_candidates = self.label2audidxs.get(lab, [])
                if not aud_candidates:
                    aud_candidates = []
                    for L in self.class_labels:
                        aud_candidates.extend(self.label2audidxs.get(L, []))
                aud_idx = random.choice(aud_candidates) if aud_candidates else None
                pairs.append((i, aud_idx))
        else:
            for j in range(self.num_audios):
                lab = self.audio_labels[j]
                img_candidates = self.label2imgidxs.get(lab, [])
                if not img_candidates:
                    img_candidates = []
                    for L in self.class_labels:
                        img_candidates.extend(self.label2imgidxs.get(L, []))
                img_idx = random.choice(img_candidates) if img_candidates else None
                pairs.append((img_idx, j))
        random.shuffle(pairs)
        self.pairs = pairs
        print(f"[Dataset] Resampled {len(self.pairs)} pairs (mode=paired).")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        self._open_files_if_needed()
        img_idx, aud_idx = self.pairs[idx]

        # read image (could be embedding vector or image tensor)
        if img_idx is None:
            # produce zeros sized like per-sample image shape (excluding N)
            meta = self.image_shape
            if len(meta) >= 2:
                vec_shape = meta[1:]
                img_np = np.zeros(vec_shape, dtype=np.float32)
            else:
                img_np = np.zeros((1,), dtype=np.float32)
        else:
            ds = self._images_h5[self.image_data_path]
            img_np = ds[img_idx]

        if isinstance(img_np, np.ndarray):
            img_np = img_np.astype(np.float32)
            # If it's HWC -> CHW convert
            if img_np.ndim == 3 and img_np.shape[-1] in (1, 3):
                img_np = np.transpose(img_np, (2, 0, 1))

        img_t = torch.from_numpy(img_np)
        if self.image_transform is not None:
            try:
                img_t = self.image_transform(img_t)
            except Exception:
                pass

        # read audio features
        if (aud_idx is None) or (not self.audio_features_present):
            aud_arr = np.zeros(self.audio_feature_shape, dtype=np.float32)
            aud_t = torch.from_numpy(aud_arr.astype(np.float32))
        else:
            aud_ds = self._audios_h5[self.audio_data_path]
            aud_np = aud_ds[aud_idx]
            try:
                aud_arr = np.array(aud_np)
                if aud_arr.dtype.kind in ('S', 'U'):
                    aud_t = torch.zeros(self.audio_feature_shape, dtype=torch.float32)
                else:
                    aud_t = torch.from_numpy(np.array(aud_np, dtype=np.float32))
            except Exception:
                aud_t = torch.zeros(self.audio_feature_shape, dtype=torch.float32)

        if self.audio_transform is not None:
            try:
                aud_t = self.audio_transform(aud_t)
            except Exception:
                pass

        lab = (self.image_labels[img_idx] if img_idx is not None else (self.audio_labels[aud_idx] if aud_idx is not None else 'unknown'))
        lab_idx = self.label2idx.get(lab, -1)
        return img_t, aud_t, lab_idx

# -------------------------
# Model supporting embeddings or images
# -------------------------
class MultimodalNet(nn.Module):
    def __init__(self, image_input_shape, audio_feat_dim, embedding_dim=256, num_classes=7):
        """
        image_input_shape: tuple including channels if image tensor, or (D,) for embeddings (no N dim).
        audio_feat_dim: int (flattened)
        """
        super().__init__()
        # Decide whether image is embedding (1D) or image tensor (C,H,W)
        if len(image_input_shape) == 1:
            # image embeddings path: build wider MLP with BN + dropout
            image_feat_dim = int(image_input_shape[0])
            self.image_is_embedding = True
            self.image_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(image_feat_dim, embedding_dim * 2),
                nn.BatchNorm1d(embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU()
            )
            image_out_dim = embedding_dim
        else:
            # treat as image tensor -> use ResNet18 backbone
            self.image_is_embedding = False
            channels = int(image_input_shape[0])
            self.image_encoder = models.resnet18(pretrained=False)
            if channels != 3:
                self.image_encoder.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nfeat = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()
            image_out_dim = nfeat

        # audio encoder: wider MLP + BN + dropout
        self.audio_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(audio_feat_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU()
        )

        # fusion head: LayerNorm + MLP
        self.fusion_norm = nn.LayerNorm(image_out_dim + embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(image_out_dim + embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, img, aud):
        img_emb = self.image_encoder(img)
        aud_emb = self.audio_encoder(aud)
        x = torch.cat([img_emb, aud_emb], dim=1)
        x = self.fusion_norm(x)
        out = self.fusion(x)
        return out

# -------------------------
# Helpers: transforms, training
# -------------------------
def default_image_transform(tensor):
    x = tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    # if it's a 1D vector leave as-is; if it's HWC convert
    if x.ndim == 3 and x.shape[0] not in (1, 3):
        x = x.permute(2, 0, 1)
    x = x.float()
    if x.max() > 20:
        x = x / 255.0
    return x


def default_audio_transform(tensor):
    return tensor.float()


def collate_fn(batch):
    imgs, auds, labs = zip(*batch)
    imgs = torch.stack(imgs)
    auds = torch.stack(auds)
    labs = torch.tensor(labs, dtype=torch.long)
    return imgs, auds, labs

def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, grad_clip: float = 0.0):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    # If criterion has a weight tensor on CPU, move it to the target device (important for MPS)
    try:
        if hasattr(criterion, 'weight') and criterion.weight is not None:
            if criterion.weight.device != device:
                criterion.weight = criterion.weight.to(device)
    except Exception:
        # don't fail on weird criterion objects
        pass

    for imgs, auds, labs in dataloader:
        imgs = imgs.to(device)
        auds = auds.to(device)
        labs = labs.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, auds)

        # ensure outputs on same device (safety)
        outputs = outputs.to(device)

        # mask must be on the same device as outputs/labs on MPS
        mask = (labs >= 0).to(device)

        if mask.sum().item() == 0:
            continue

        loss = criterion(outputs[mask], labs[mask])
        loss.backward()

        if grad_clip and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # step scheduler per batch if provided (OneCycleLR expects per-batch stepping)
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        running_loss += loss.item() * mask.sum().item()
        _, preds = outputs[mask].max(1)
        correct += (preds == labs[mask]).sum().item()
        total += mask.sum().item()

    avg_loss = running_loss / (total + 1e-8)
    acc = correct / (total + 1e-8)
    return avg_loss, acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    # move criterion weight to device if needed
    try:
        if hasattr(criterion, 'weight') and criterion.weight is not None:
            if criterion.weight.device != device:
                criterion.weight = criterion.weight.to(device)
    except Exception:
        pass

    with torch.no_grad():
        for imgs, auds, labs in dataloader:
            imgs = imgs.to(device)
            auds = auds.to(device)
            labs = labs.to(device)

            outputs = model(imgs, auds)
            outputs = outputs.to(device)

            mask = (labs >= 0).to(device)
            if mask.sum().item() == 0:
                continue

            loss = criterion(outputs[mask], labs[mask])
            running_loss += loss.item() * mask.sum().item()
            _, preds = outputs[mask].max(1)
            correct += (preds == labs[mask]).sum().item()
            total += mask.sum().item()

    avg_loss = running_loss / (total + 1e-8)
    acc = correct / (total + 1e-8)
    return avg_loss, acc


def get_compute_device_name_and_obj():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "cuda"
        return f"cuda:{name}", torch.device("cuda")
    return "cpu", torch.device("cpu")

# -------------------------
# Helper: compute class weights from train dataset (handles Subset)
# -------------------------
def compute_class_weights(train_dataset, ds_ref, class_labels):
    """
    Returns a torch.FloatTensor of weights ordered by class_labels.
    train_dataset may be a Subset; ds_ref is the underlying EmotionPairDataset (or dataset)
    """
    # Extract label list for examples included in train_dataset
    if hasattr(train_dataset, 'indices'):
        # Subset: use indices to sample ds_ref.image_labels
        indices = train_dataset.indices
        labels = [ds_ref.image_labels[i] for i in indices]
    else:
        labels = ds_ref.image_labels

    # map labels to only those in class_labels (should be)
    counts = Counter([str(l) for l in labels if str(l) in class_labels])
    # fallback: if counts missing classes, add small epsilon
    freqs = [counts.get(c, 0) for c in class_labels]
    # avoid zero division
    freqs = [f if f > 0 else 1 for f in freqs]
    inv = [1.0 / float(f) for f in freqs]
    # normalize to mean 1
    mean_inv = float(np.mean(inv))
    weights = [w / mean_inv for w in inv]
    return torch.tensor(weights, dtype=torch.float)

# -------------------------
# Main / CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_h5', type=str,
                        default='emotion_service/saved_paired_train_paired_embeddings.h5')
    parser.add_argument('--val_h5', type=str,
                        default='emotion_service/saved_paired_val_paired_embeddings.h5')
    parser.add_argument('--test_h5', type=str,
                        default='emotion_service/saved_paired_test_paired_embeddings.h5')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_lr', type=float, default=2e-4, help="max_lr for OneCycleLR (per-batch)")
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefer_pairs_base', choices=['images', 'audios'], default='images')
    parser.add_argument('--save_model', type=str, default='multimodal_best.pth')
    parser.add_argument('--save_dir', type=str, default='emotion_service/saved_models')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_scheduler', action='store_true', help="Enable OneCycleLR scheduler (stepped per batch).")
    parser.add_argument('--use_cosine', action='store_true', help="Use CosineAnnealingLR (epoch-level) instead of OneCycleLR when --use_scheduler is not set.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    image_transform = default_image_transform
    audio_transform = default_audio_transform

    if not os.path.exists(args.train_h5):
        raise RuntimeError(f"Train HDF5 file not found: {args.train_h5}")

    # build datasets
    train_dataset_full = EmotionPairDataset(args.train_h5, args.train_h5,
                                           image_transform=image_transform,
                                           audio_transform=audio_transform,
                                           prefer_pairs_base=args.prefer_pairs_base)

    if os.path.exists(args.val_h5):
        val_dataset = EmotionPairDataset(args.val_h5, args.val_h5,
                                         image_transform=image_transform,
                                         audio_transform=audio_transform,
                                         prefer_pairs_base=args.prefer_pairs_base)
        train_dataset = train_dataset_full
    else:
        # fallback 80/20 split
        n = len(train_dataset_full)
        if n < 2:
            raise RuntimeError("Dataset too small.")
        idxs = list(range(n))
        random.shuffle(idxs)
        split = int(0.8 * n)
        train_idx = idxs[:split]
        val_idx = idxs[split:]
        train_dataset = Subset(train_dataset_full, train_idx)
        val_dataset_full = EmotionPairDataset(args.train_h5, args.train_h5,
                                             image_transform=image_transform,
                                             audio_transform=audio_transform,
                                             prefer_pairs_base=args.prefer_pairs_base)
        val_dataset = Subset(val_dataset_full, val_idx)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    try:
        sample_ref = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
        print(f"Class labels (sample): {getattr(sample_ref, 'class_labels', None)}")
    except Exception:
        pass

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, persistent_workers=(args.workers>0))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn, persistent_workers=(args.workers>0))

    dev_name, device = get_compute_device_name_and_obj()
    print(f"[device] Running training on: {dev_name} -> {device}")

    # infer shapes for encoders
    ds_ref = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
    img_shape = getattr(ds_ref, 'image_shape', None)
    aud_feat_shape = getattr(ds_ref, 'audio_feature_shape', None)

    # If image is embedding (N, D) -> image_input_shape = (D,)
    if img_shape is None:
        # get sample batch
        try:
            sample_imgs, sample_auds, _ = next(iter(train_loader))
            image_input_shape = tuple(sample_imgs.shape[1:])
            aud_shape_sample = tuple(sample_auds.shape[1:])
            audio_feat_dim = int(np.prod(aud_shape_sample)) if len(aud_shape_sample) > 0 else 1
        except Exception:
            image_input_shape = (3, 224, 224)
            audio_feat_dim = 128
    else:
        if len(img_shape) == 2:
            image_input_shape = (int(img_shape[1]),)
        elif len(img_shape) >= 3:
            # derive channels-first if necessary: handle (N, C, H, W) or (N, H, W, C)
            if len(img_shape) == 4 and img_shape[1] in (1,3):
                image_input_shape = (int(img_shape[1]), int(img_shape[2]), int(img_shape[3]))
            elif len(img_shape) == 4 and img_shape[3] in (1,3):
                image_input_shape = (int(img_shape[3]), int(img_shape[1]), int(img_shape[2]))
            elif len(img_shape) == 3:
                # ambiguous: assume (N, D1, D2) -> treat as (C,H,W) with C=img_shape[1]? fallback to (3,H,W)
                if img_shape[2] in (1,3):
                    image_input_shape = (int(img_shape[2]), int(img_shape[1]), 1)
                else:
                    image_input_shape = (3, int(img_shape[1]), int(img_shape[2]))
            else:
                image_input_shape = (3, 224, 224)
        else:
            image_input_shape = (3, 224, 224)

        if aud_feat_shape is None:
            audio_feat_dim = 128
        else:
            audio_feat_dim = int(np.prod(aud_feat_shape)) if len(aud_feat_shape) > 0 else 1

    print(f"[model] image_input_shape={image_input_shape} audio_feat_dim={audio_feat_dim}")

    model = MultimodalNet(image_input_shape=image_input_shape, audio_feat_dim=audio_feat_dim,
                          embedding_dim=256, num_classes=len(getattr(ds_ref, 'class_labels', [0])))
    model.to(device)

    # compute class weights and build criterion
    class_labels = getattr(ds_ref, 'class_labels', None)
    if class_labels is None:
        criterion = nn.CrossEntropyLoss()
    else:
        weights = compute_class_weights(train_dataset, ds_ref, class_labels)
        # CrossEntropyLoss expects weight on CPU; it's fine to keep on CPU
        criterion = nn.CrossEntropyLoss(weight=weights)

    # optimizer and scheduler (OneCycleLR per-batch)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_scheduler:
        try:
            steps_per_epoch = max(1, len(train_loader))
            total_steps = int(args.epochs * steps_per_epoch)
            # choose max_lr from arg
            scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps, pct_start=0.1)
            print(f"[scheduler] OneCycleLR enabled: total_steps={total_steps}, max_lr={args.max_lr}")
        except Exception as e:
            print(f"[scheduler] could not create OneCycleLR: {e}. Continuing without per-batch scheduler.")
            scheduler = None
    elif args.use_cosine:
        try:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
            print("[scheduler] CosineAnnealingLR will be stepped per-epoch.")
        except Exception as e:
            print(f"[scheduler] could not create CosineAnnealingLR: {e}. Continuing without scheduler.")
            scheduler = None

    best_val_acc = 0.0
    best_ckpt = None
    for epoch in range(1, args.epochs + 1):
        st = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}: resampling dataset pairs ...")
        try:
            if hasattr(train_dataset, 'dataset'):
                train_dataset.dataset.resample_pairs()
            else:
                train_dataset.resample_pairs()
        except Exception:
            pass

        # if using epoch-level scheduler (Cosine), we will step after validation
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler=(scheduler if args.use_scheduler else None), grad_clip=args.grad_clip)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # if using epoch-level scheduler (CosineAnnealingLR), step once per epoch
        if scheduler is not None and args.use_cosine:
            try:
                scheduler.step()
            except Exception:
                pass

        elapsed = time.time() - st
        print(f"Epoch {epoch} done in {elapsed:.1f}s — Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, args.save_model)
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'class_labels': getattr(ds_ref, 'class_labels', None),
                }, save_path)
                best_ckpt = save_path
                print(f"Saved best model to {save_path} (val_acc={val_acc:.4f})")
            except Exception as e:
                print(f"ERROR: failed to save model to {save_path}: {e}")

    print("Training complete. Best val acc:", best_val_acc)

    if os.path.exists(args.test_h5) and best_ckpt is not None:
        print(f"Evaluating best checkpoint on test set: {args.test_h5}")
        try:
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        except Exception as e:
            print(f"Could not load checkpoint for testing: {e}")
            return
        test_ds = EmotionPairDataset(args.test_h5, args.test_h5,
                                     image_transform=image_transform,
                                     audio_transform=audio_transform,
                                     prefer_pairs_base=args.prefer_pairs_base)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Test set — loss: {test_loss:.4f}, acc: {test_acc:.4f}")
    elif os.path.exists(args.test_h5):
        print("Test H5 exists but no checkpoint was saved during training to evaluate.")

if __name__ == '__main__':
    main()

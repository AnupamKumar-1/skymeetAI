#!/usr/bin/env python3
"""
Evaluation & plotting script for multimodal emotion model.

This script loads a saved checkpoint (saved with train_multimodal.py), builds the
same MultimodalNet architecture automatically by inspecting a sample from the
H5 dataset, runs inference on test (and optionally validation) sets, computes
metrics (accuracy, F1, classification report, confusion matrix) and saves plots
(to PNG files) into the save directory.

Usage example:
    python emotion_service/train_eval_plot.py \
        --model_path emotion_service/saved_models/multimodal_best.pth \
        --test_h5 data/test.h5 \
        --val_h5 data/val.h5 \
        --save_dir results/plots

Notes:
- This script imports helper classes from train_multimodal.py (EmotionPairDataset,
  collate_fn, MultimodalNet, default transforms, validate). Keep train_multimodal.py
  next to this script.
- If --val_h5 is provided the script will compute val_acc and plot Test vs Val.
- If sklearn is not available the script will still compute basic accuracy but
  will warn and skip classification_report/confusion matrix plotting.

"""

import os
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Import constructs from train_multimodal.py (must be on PYTHONPATH / same folder)
try:
    from training import train_multimodal as tm
except Exception as e:
    raise RuntimeError(
        "Could not import train_multimodal.py from the current directory. "
        "Make sure this script sits next to train_multimodal.py or that it is importable. "
        f"Import error: {e}"
    )

# Helpers for plotting & metrics
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def build_model_from_sample(sample_img, sample_aud, num_classes: int, embedding_dim: int = 256):
    """Infer shapes from one sample and construct MultimodalNet."""
    # sample tensors may be numpy or torch tensors
    if not isinstance(sample_img, torch.Tensor):
        sample_img = torch.as_tensor(sample_img)
    if not isinstance(sample_aud, torch.Tensor):
        sample_aud = torch.as_tensor(sample_aud)

    # image_input_shape: tuple (channels, H, W) or (D,)
    image_input_shape = tuple(sample_img.shape)
    # audio_feat_dim: flattened length
    audio_feat_dim = int(sample_aud.numel())

    model = tm.MultimodalNet(image_input_shape=image_input_shape,
                             audio_feat_dim=audio_feat_dim,
                             embedding_dim=embedding_dim,
                             num_classes=num_classes)
    return model


def evaluate_and_collect(model, dataloader, device):
    model.eval()
    preds = []
    trues = []
    probs = []
    with torch.no_grad():
        for imgs, auds, labs in dataloader:
            imgs = imgs.to(device)
            auds = auds.to(device)
            labs = labs.to(device)
            outputs = model(imgs, auds)
            # outputs: (N, num_classes)
            softmax = torch.softmax(outputs, dim=1)
            _, pred = outputs.max(1)
            preds.append(pred.cpu())
            trues.append(labs.cpu())
            probs.append(softmax.cpu())
    if len(preds) == 0:
        return None, None, None
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    probs = torch.cat(probs).numpy()
    # filter out masked/unknown labels (<0)
    mask = trues >= 0
    return preds[mask], trues[mask], probs[mask]


def plot_confusion_matrix(cm, labels, outpath, normalize=False):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm_to_plot = cm.astype('float') / (cm.sum(axis=1)[:, None] + 1e-8)
    else:
        cm_to_plot = cm
    im = plt.imshow(cm_to_plot, interpolation='nearest', aspect='auto')
    plt.title('Confusion matrix' + (' (normalized)' if normalize else ''))
    plt.colorbar(im)
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_test_vs_val(test_acc: float, val_acc: Optional[float], outpath: str):
    plt.figure(figsize=(4, 4))
    if val_acc is None:
        plt.bar(['test'], [test_acc])
        plt.ylim(0, 1)
        plt.title('Test accuracy')
    else:
        plt.bar(['validation', 'test'], [val_acc, test_acc])
        plt.ylim(0, 1)
        plt.title('Validation vs Test accuracy')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--test_h5', required=True, help='H5 file for test set (compatible with train_multimodal)')
    parser.add_argument('--val_h5', default=None, help='Optional H5 file for validation set to compute val_acc')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', default='plots')
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--device', default=None, help='torch device (e.g. cpu or cuda)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('Using device:', device)

    # Build dataset(s)
    image_transform = tm.default_image_transform
    audio_transform = tm.default_audio_transform

    test_ds = tm.EmotionPairDataset(args.test_h5, args.test_h5,
                                    image_transform=image_transform,
                                    audio_transform=audio_transform,
                                    prefer_pairs_base='images')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, collate_fn=tm.collate_fn)

    val_loader = None
    if args.val_h5:
        val_ds = tm.EmotionPairDataset(args.val_h5, args.val_h5,
                                       image_transform=image_transform,
                                       audio_transform=audio_transform,
                                       prefer_pairs_base='images')
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, collate_fn=tm.collate_fn)

    # Load checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)

    # Determine class labels
    class_labels = None
    if 'class_labels' in ckpt and ckpt['class_labels'] is not None:
        class_labels = ckpt['class_labels']
    elif hasattr(test_ds, 'class_labels') and getattr(test_ds, 'class_labels') is not None:
        class_labels = test_ds.class_labels
    else:
        # fallback: use integer labels 0..C-1 by inspecting checkpoint if present
        print('Warning: could not determine class label names. Using numeric labels.')

    # If class_labels is still None, try to infer number of classes from model_state_dict final linear
    if class_labels is None:
        # try to infer num_classes from checkpoint model_state_dict keys
        msd = ckpt.get('model_state_dict', {})
        num_classes = None
        for k, v in msd.items():
            if 'fusion' in k and 'weight' in k:
                # last linear's weight shape is (num_classes, hidden)
                num_classes = v.shape[0]
                break
        if num_classes is None:
            # try dataset labels
            num_classes = int(getattr(test_ds, 'num_classes', -1))
            if num_classes <= 0:
                raise RuntimeError('Could not infer number of classes. Provide model with class_labels or dataset with num_classes.')
        class_labels = [str(i) for i in range(num_classes)]
    else:
        class_labels = list(class_labels)
        num_classes = len(class_labels)

    # Use a sample to build the model
    sample_img, sample_aud, _ = test_ds[0]
    model = build_model_from_sample(sample_img, sample_aud, num_classes=num_classes, embedding_dim=args.embedding_dim)
    model.to(device)

    # load state dict
    model_state = ckpt.get('model_state_dict', ckpt)
    try:
        model.load_state_dict(model_state)
    except Exception as e:
        # attempt to allow for module prefix differences
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in model_state.items():
            new_key = k.replace('module.', '')
            new_state[new_key] = v
        model.load_state_dict(new_state)

    # Evaluate on test
    preds, trues, probs = evaluate_and_collect(model, test_loader, device)
    if preds is None:
        print('No predictions were generated on test set (empty?). Exiting.')
        return

    test_acc = (preds == trues).mean()
    print(f'Test accuracy: {test_acc:.4f}')

    # Compute F1
    if SKLEARN_AVAILABLE:
        f1_macro = f1_score(trues, preds, average='macro')
        f1_micro = f1_score(trues, preds, average='micro')
        print(f'F1 macro: {f1_macro:.4f}, F1 micro: {f1_micro:.4f}')

        cr = classification_report(trues, preds, target_names=class_labels, zero_division=0)
        print('Classification report:\n', cr)
    else:
        print('scikit-learn not available: skipping F1 and classification report.')

    # Confusion matrix
    if SKLEARN_AVAILABLE:
        cm = confusion_matrix(trues, preds, labels=list(range(num_classes)))
        cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, class_labels, cm_path, normalize=False)
        cm_norm_path = os.path.join(args.save_dir, 'confusion_matrix_normalized.png')
        plot_confusion_matrix(cm, class_labels, cm_norm_path, normalize=True)
        print('Saved confusion matrix plots to', args.save_dir)
    else:
        print('sklearn unavailable: cannot compute confusion matrix plot.')

    # Optionally evaluate on validation set
    val_acc = None
    if val_loader is not None:
        vpreds, vtrues, vprobs = evaluate_and_collect(model, val_loader, device)
        if vpreds is not None:
            val_acc = (vpreds == vtrues).mean()
            print(f'Validation accuracy: {val_acc:.4f}')

    # Save test vs val bar chart
    tv_path = os.path.join(args.save_dir, 'test_vs_val_accuracy.png')
    plot_test_vs_val(test_acc, val_acc, tv_path)
    print('Saved test vs val plot to', tv_path)

    # Save per-class F1 bar chart if sklearn present
    if SKLEARN_AVAILABLE:
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f1s, sup = precision_recall_fscore_support(trues, preds, labels=list(range(num_classes)), zero_division=0)
        plt.figure(figsize=(10, 4))
        plt.bar(range(num_classes), f1s)
        plt.xticks(range(num_classes), class_labels, rotation=45, ha='right')
        plt.ylabel('F1')
        plt.title('Per-class F1 scores')
        plt.tight_layout()
        outp = os.path.join(args.save_dir, 'per_class_f1.png')
        plt.savefig(outp, dpi=200)
        plt.close()
        print('Saved per-class F1 plot to', outp)

    # Also save raw predictions & labels for downstream analysis
    import json
    preds_path = os.path.join(args.save_dir, 'predictions_and_labels.npz')
    import numpy as _np
    _np.savez_compressed(preds_path, preds=preds, trues=trues, probs=probs)
    print('Saved raw predictions to', preds_path)


if __name__ == '__main__':
    main()

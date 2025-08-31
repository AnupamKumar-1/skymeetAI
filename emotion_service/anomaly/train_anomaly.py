#!/usr/bin/env python3
"""
Train an Isolation Forest anomaly detector on paired embeddings H5 files.

This script will:
- scan the provided H5 for image/audio embedding datasets (auto-detect by name hints)
- load training embeddings (concatenate image+audio if both present)
- fit sklearn.ensemble.IsolationForest
- save the trained model (joblib) and arrays of anomaly scores
- optionally score a test H5 and save plots (score histograms, PCA scatter)

Example:
    python anomaly/train_anomaly.py \
        --train_h5 ../saved_paired_train_paired_embeddings.h5 \
        --test_h5 ../saved_paired_test_paired_embeddings.h5 \
        --save_dir ../results/anomaly_iforest \
        --n_estimators 200 --contamination 0.01
"""
import os
import argparse
import h5py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt


def find_embedding_paths(h5_path):
    """Return (image_path, audio_path, label_path) or None if not found."""
    img_path = None
    aud_path = None
    label_path = None
    tokens_image = ['image', 'img', 'face', 'visual', 'image_embedding', 'image_emb', 'emb', 'embedding']
    tokens_audio = ['audio', 'aud', 'speech', 'wav', 'mel', 'mfcc', 'audio_embedding', 'audio_emb', 'feat', 'feature']
    tokens_label = ['label', 'labels', 'y', 'class']

    with h5py.File(h5_path, 'r') as f:
        datasets = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)
        f.visititems(visitor)

    # score candidates by token matches
    for d in datasets:
        low = d.lower()
        for t in tokens_image:
            if t in low:
                img_path = d
                break
        for t in tokens_audio:
            if t in low:
                aud_path = d
                break
        for t in tokens_label:
            if t in low:
                label_path = d
                break
    # fallback heuristics
    if img_path is None:
        # pick highest-dim dataset (prefer 2D+) as image
        best = None
        best_rank = -1
        with h5py.File(h5_path, 'r') as f:
            for d in datasets:
                shape = f[d].shape
                rank = len(shape)
                if rank >= 2 and rank > best_rank:
                    best = d
                    best_rank = rank
        img_path = best
    return img_path, aud_path, label_path


def load_embeddings(h5_path, img_path=None, aud_path=None):
    """Load embeddings from H5 and return (X, maybe labels)
    X is (N, D) numpy array.
    """
    with h5py.File(h5_path, 'r') as f:
        if img_path is None and aud_path is None:
            raise RuntimeError('No image or audio embedding dataset paths provided/found')
        img_arr = None
        aud_arr = None
        if img_path is not None:
            img_arr = np.array(f[img_path])
            # ensure (N, D)
            if img_arr.ndim > 2:
                # flatten per sample
                img_arr = img_arr.reshape((img_arr.shape[0], -1))
        if aud_path is not None:
            aud_arr = np.array(f[aud_path])
            if aud_arr.ndim > 2:
                aud_arr = aud_arr.reshape((aud_arr.shape[0], -1))

        if img_arr is not None and aud_arr is not None:
            if img_arr.shape[0] != aud_arr.shape[0]:
                raise RuntimeError('Image and audio embeddings have different number of samples')
            X = np.concatenate([img_arr, aud_arr], axis=1)
        elif img_arr is not None:
            X = img_arr
        else:
            X = aud_arr

        # attempt to load labels if present (optional)
        labels = None
        # check for common label datasets near root
        for cand in ['labels', 'label', 'y', 'class_labels', 'class_label']:
            if cand in f:
                try:
                    labels = np.array(f[cand])
                except Exception:
                    labels = None
                break

    return X, labels


def plot_score_hist(train_scores, test_scores, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(train_scores, bins=60, alpha=0.7, label='train', density=True)
    if test_scores is not None:
        plt.hist(test_scores, bins=60, alpha=0.7, label='test', density=True)
    plt.legend()
    plt.title('Anomaly score distribution (higher = more anomalous)')
    plt.tight_layout()
    path = os.path.join(outdir, 'anomaly_score_hist.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print('Saved', path)


def plot_pca_scatter(X, scores, outdir, name_prefix='train'):
    os.makedirs(outdir, exist_ok=True)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=scores, s=6)
    plt.colorbar(label='anomaly score (higher = more anomalous)')
    plt.title(f'PCA scatter ({name_prefix})')
    plt.tight_layout()
    path = os.path.join(outdir, f'pca_scatter_{name_prefix}.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print('Saved', path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_h5', required=True)
    parser.add_argument('--test_h5', default=None)
    parser.add_argument('--save_dir', default='saved_models')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--contamination', type=float, default=0.01)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--eval_with_labels', action='store_true', help='If set, try a simple binary eval when labels look binary (0/1)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # detect paths
    print('[paired-scanner] scanning train file', args.train_h5)
    train_img, train_aud, _ = find_embedding_paths(args.train_h5)
    print('[paired-scanner] chosen image dataset', train_img)
    print('[paired-scanner] chosen audio dataset', train_aud)

    X_train, train_labels = load_embeddings(args.train_h5, img_path=train_img, aud_path=train_aud)
    print('[Dataset] loaded train features shape:', X_train.shape)

    X_test = None
    test_labels = None
    if args.test_h5:
        print('[paired-scanner] scanning test file', args.test_h5)
        test_img, test_aud, _ = find_embedding_paths(args.test_h5)
        print('[paired-scanner] chosen image dataset', test_img)
        print('[paired-scanner] chosen audio dataset', test_aud)
        X_test, test_labels = load_embeddings(args.test_h5, img_path=test_img, aud_path=test_aud)
        print('[Dataset] loaded test features shape:', X_test.shape)

    # fit IsolationForest
    clf = IsolationForest(n_estimators=args.n_estimators, contamination=args.contamination, random_state=args.random_state)
    print('[model] fitting IsolationForest...')
    clf.fit(X_train)
    model_path = os.path.join(args.save_dir, 'isolation_forest.joblib')
    joblib.dump(clf, model_path)
    print('[model] saved to', model_path)

    # compute anomaly scores (higher = more anomalous)
    # sklearn's decision_function: higher = more normal, so negate it
    train_scores = -clf.decision_function(X_train)
    test_scores = -clf.decision_function(X_test) if X_test is not None else None

    # predicted anomalies based on contamination (or predict method)
    train_pred = (clf.predict(X_train) == -1).astype(int)
    test_pred = (clf.predict(X_test) == -1).astype(int) if X_test is not None else None

    # print helpful summary statistics to terminal
    def summarize_scores(name, scores, preds=None, top_k=10):
        if scores is None:
            print(f'No scores for {name}')
            return
        scores = np.asarray(scores)
        n = len(scores)
        mean = float(np.mean(scores))
        med = float(np.median(scores))
        std = float(np.std(scores))
        mn = float(np.min(scores))
        mx = float(np.max(scores))
        # threshold implied by contamination: select top contamination fraction as anomalies
        try:
            thresh = np.quantile(scores, 1.0 - args.contamination)
        except Exception:
            thresh = None
        n_anom_pred = int((preds == 1).sum()) if preds is not None else (int((scores >= thresh).sum()) if thresh is not None else None)

        # Print summary with a leading newline for readability
        print(f"\n--- Anomaly score summary for {name} ---")
        print(f'  samples: {n}')
        print(f'  mean: {mean:.6f}, median: {med:.6f}, std: {std:.6f}')
        print(f'  min: {mn:.6f}, max: {mx:.6f}')
        if thresh is not None:
            print(f'  contamination threshold (top {args.contamination*100:.3g}%) score >= {thresh:.6f}')
        if n_anom_pred is not None:
            print(f'  predicted anomalies (by model): {n_anom_pred} ({(n_anom_pred/n*100):.3f}%)')
        # top-k anomalies
        top_idx = np.argsort(-scores)[:top_k]
        print(f'  top {top_k} anomaly scores (index: score):')
        for i in top_idx:
            print(f'    {i}: {scores[i]:.6f}')

    summarize_scores('train', train_scores, train_pred)
    summarize_scores('test', test_scores, test_pred)

    # save raw outputs
    out_npz = os.path.join(args.save_dir, 'anomaly_scores.npz')
    np.savez_compressed(out_npz, train_scores=train_scores, train_pred=train_pred, test_scores=test_scores, test_pred=test_pred)
    print('Saved scores to', out_npz)

    # plots
    plot_score_hist(train_scores, test_scores, args.save_dir)
    plot_pca_scatter(X_train, train_scores, args.save_dir, name_prefix='train')
    if X_test is not None:
        plot_pca_scatter(X_test, test_scores, args.save_dir, name_prefix='test')

    # optional simple evaluation if binary labels present
    if args.eval_with_labels:
        try:
            import sklearn.metrics as skm
            # determine if labels are binary (0/1 or -1/1)
            labels = None
            if test_labels is not None:
                labels = test_labels
            elif train_labels is not None:
                labels = train_labels
            if labels is not None:
                # try to normalize labels to 0/1
                lab_vals = np.unique(labels)
                lab_set = set(lab_vals.tolist())
                if lab_set <= {0, 1} or lab_set <= {-1, 1}:
                    if lab_set <= {-1, 1}:
                        y_true = (labels == 1).astype(int)
                    else:
                        y_true = labels.astype(int)
                    y_pred = test_pred if test_pred is not None else train_pred
                    if y_pred is not None and len(y_pred) == len(y_true):
                        # Use anomaly score (higher = more anomalous) as decision function for AUC
                        if test_scores is not None and len(test_scores) == len(y_true):
                            auc = skm.roc_auc_score(y_true, test_scores)
                            print('AUC (test scores vs labels):', auc)
                        else:
                            auc = skm.roc_auc_score(y_true, train_scores[:len(y_true)])
                            print('AUC (train scores vs labels):', auc)

                        print("\nClassification report (treating predicted anomalies as positive class=1):")
                        print(skm.classification_report(y_true, y_pred, zero_division=0))
                    else:
                        print('Could not match predictions to provided labels for evaluation')
                else:
                    print('Found labels but they do not look binary; skipping binary evaluation')
            else:
                print('No labels found in H5 for evaluation')
        except Exception as e:
            print('Evaluation requested but sklearn.metrics not available or failed:', e)

    print('Done.')


if __name__ == '__main__':
    main()

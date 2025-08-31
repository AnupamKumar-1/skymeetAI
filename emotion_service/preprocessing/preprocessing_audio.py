#!/usr/bin/env python3
"""Preprocess audio into HDF5 features + CSV manifest.

Usage examples (from project root emotion_service):
  python preprocessing/preprocessing_audio_h5.py --audio_dir data/audio
  python preprocessing/preprocessing_audio_h5.py --audio_dir data/audio --out_h5 preprocessed_audio.h5 --fixed_duration 3.0 --workers 6

Notes:
 - If --fixed_duration > 0, features are fixed-shape (fastest). If you set --fixed_duration 0, variable-length features will be stored.
 - Uses ProcessPoolExecutor for CPU-bound work; HDF5 writes are performed by the main process only.
Dependencies:
  pip install librosa soundfile numpy pandas tqdm h5py
"""
from pathlib import Path
import argparse
import sys
import os
import numpy as np
import pandas as pd
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import h5py
import traceback

# Emotion extraction mapping guessed from CREMA-D filename codes
EMOTION_MAP = {
    'ANG': 'anger',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'SAD': 'sad',
    'NEU': 'neutral'
}


def extract_label_from_filename(filename: str) -> str:
    up = filename.upper()
    for code, label in EMOTION_MAP.items():
        if code in up:
            return label
    return 'unknown'


def _process_file_worker(args):
    """
    Worker function to be executed in another process.
    Returns a tuple: (ok_bool, result_dict)
    If ok_bool True -> result contains {'feature': np.ndarray, 'frames': int (if variable), 'meta': {...}}
    If ok_bool False -> result contains {'original_path': str, 'status': 'error_*', 'error': str}
    """
    (path_str, sr, n_mels, n_fft, hop_length, fixed_duration, top_db, pad_mode) = args
    path = Path(path_str)
    try:
        y, orig_sr = librosa.load(path.as_posix(), sr=sr, mono=True)
    except Exception as e:
        return False, {'original_path': str(path), 'status': 'error_load', 'error': str(e)}

    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    except Exception:
        y_trimmed = y

    if fixed_duration is not None and fixed_duration > 0:
        target_len = int(round(fixed_duration * sr))
        if len(y_trimmed) > target_len:
            start = max(0, (len(y_trimmed) - target_len) // 2)
            y_fixed = y_trimmed[start:start + target_len]
        else:
            pad_len = target_len - len(y_trimmed)
            left = pad_len // 2
            right = pad_len - left
            y_fixed = np.pad(y_trimmed, (left, right), mode=pad_mode)
    else:
        y_fixed = y_trimmed

    peak = np.max(np.abs(y_fixed)) if len(y_fixed) > 0 else 0.0
    if peak > 0:
        y_fixed = y_fixed / float(max(1e-8, peak))

    try:
        mel = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_fft=n_fft,
                                             hop_length=hop_length, n_mels=n_mels, power=2.0)
        log_mel = librosa.power_to_db(mel, ref=np.max)
    except Exception as e:
        return False, {'original_path': str(path), 'status': 'error_mel', 'error': str(e)}

    # metadata
    label = extract_label_from_filename(path.name)
    meta = {
        'original_path': str(path),
        'sr': sr,
        'duration_sec': round(len(y_fixed) / float(sr), 6),
        'n_mels': n_mels,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'label': label,
        'status': 'ok'
    }

    # Return array and meta
    return True, {'feature': log_mel.astype(np.float32), 'frames': log_mel.shape[1], 'meta': meta}


def find_audio_files(audio_dir: Path, exts=('.wav', '.WAV', '.flac', '.FLAC')):
    for p in audio_dir.rglob('*'):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def create_fixed_dset(grp, name, n_mels, frames, chunk_n=16, compression=None, dtype=np.float32):
    maxshape = (None, n_mels, frames)
    chunk = (min(chunk_n, 16), n_mels, frames)
    return grp.create_dataset(name, shape=(0, n_mels, frames), maxshape=maxshape, chunks=chunk,
                              dtype=dtype, compression=compression)


def append_fixed_dset(dset, arr_batch):
    """arr_batch: list/array of shape (B, n_mels, frames)"""
    if len(arr_batch) == 0:
        return
    cur = dset.shape[0]
    B = len(arr_batch)
    dset.resize(cur + B, axis=0)
    dset[cur:cur + B, ...] = np.stack(arr_batch, axis=0)


def create_vlen_dset(grp, name, compression=None):
    vlen_dt = h5py.vlen_dtype(np.dtype('float32'))
    return grp.create_dataset(name, shape=(0,), maxshape=(None,), dtype=vlen_dt, chunks=(256,), compression=compression)


def append_vlen_dset(dset, flat_batch):
    if len(flat_batch) == 0:
        return
    cur = dset.shape[0]
    B = len(flat_batch)
    dset.resize(cur + B, axis=0)
    dset[cur:cur + B] = flat_batch


def main():
    parser = argparse.ArgumentParser(description='Preprocess audio and store features in HDF5')
    parser.add_argument('--audio_dir', required=True, help='Root folder containing audio files')
    parser.add_argument('--out_h5', required=False, default=None, help='Output HDF5 file path (default: ./preprocessed_audio.h5)')
    parser.add_argument('--out_dir', required=False, default=None, help='Also save manifest CSV to this folder (defaults to project preprocessed_audio)')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--n_mels', type=int, default=64, help='Number of mel bands')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for STFT')
    parser.add_argument('--fixed_duration', type=float, default=3.0, help='Pad/truncate each clip to this many seconds (set 0 to disable fixed-size features)')
    parser.add_argument('--top_db', type=int, default=20, help='Top dB for trimming silence')
    parser.add_argument('--pad_mode', type=str, default='constant', choices=['constant', 'reflect', 'wrap'], help='Padding mode')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--manifest_name', type=str, default='manifest.csv', help='CSV manifest filename')
    parser.add_argument('--batch_write', type=int, default=64, help='Write to HDF5 in batches of this size')
    parser.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'lzf', 'none'], help='HDF5 compression')
    parser.add_argument('--compression_level', type=int, default=4, help='gzip level (1-9)')
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).expanduser().resolve()
    if not audio_dir.exists():
        print(f'ERROR: audio_dir not found: {audio_dir}', file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_out_root = project_root / 'preprocessed_audio'
    out_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_root
    out_root.mkdir(parents=True, exist_ok=True)

    out_h5 = Path(args.out_h5).expanduser().resolve() if args.out_h5 else (out_root / 'preprocessed_audio.h5')

    files = list(find_audio_files(audio_dir))
    if len(files) == 0:
        print(f'No audio files found in {audio_dir}', file=sys.stderr)
        sys.exit(1)

    compression = None if args.compression == 'none' else args.compression
    comp_opts = {'compression_opts': args.compression_level} if compression == 'gzip' else {}

    print(f'Found {len(files)} audio files. Writing HDF5 -> {out_h5}')
    print(f'Params: sr={args.sr}, n_mels={args.n_mels}, n_fft={args.n_fft}, hop={args.hop_length}, fixed_duration={args.fixed_duration}')

    # Prepare worker arguments
    worker_inputs = []
    for p in files:
        worker_inputs.append((str(p), args.sr, args.n_mels, args.n_fft, args.hop_length,
                              args.fixed_duration if args.fixed_duration > 0 else None,
                              args.top_db, args.pad_mode))

    # Open HDF5 for writing
    with h5py.File(out_h5, 'w') as h5f:
        # store metadata
        h5f.attrs['audio_dir'] = str(audio_dir)
        h5f.attrs['sr'] = args.sr
        h5f.attrs['n_mels'] = args.n_mels
        h5f.attrs['n_fft'] = args.n_fft
        h5f.attrs['hop_length'] = args.hop_length
        h5f.attrs['fixed_duration'] = args.fixed_duration
        h5f.attrs['pad_mode'] = args.pad_mode

        grp = h5f.require_group('features')

        manifest_rows = []
        errors = []

        # create datasets lazily depending on fixed vs variable
        fixed_mode = (args.fixed_duration is not None and args.fixed_duration > 0)
        fixed_frames = None
        if fixed_mode:
            # compute frames for fixed_duration
            target_len = int(round(args.fixed_duration * args.sr))
            # number of frames = 1 + floor((target_len - n_fft) / hop_length) [librosa STFT framing],
            # but more robust is to compute using mel shape from a dummy array
            dummy = np.zeros(target_len, dtype=np.float32)
            mel = librosa.feature.melspectrogram(y=dummy, sr=args.sr, n_fft=args.n_fft,
                                                 hop_length=args.hop_length, n_mels=args.n_mels, power=2.0)
            fixed_frames = mel.shape[1]
            # create fixed dataset
            features_dset = create_fixed_dset(grp, 'fixed', args.n_mels, fixed_frames,
                                              chunk_n=max(1, min(args.batch_write, 16)),
                                              compression=compression, dtype=np.float32)
            labels_dset = grp.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype('utf-8'),
                                             chunks=(256,), compression=compression, **(comp_opts if compression == 'gzip' else {}))
            paths_dset = grp.create_dataset('paths', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype('utf-8'),
                                            chunks=(256,), compression=compression, **(comp_opts if compression == 'gzip' else {}))
        else:
            features_dset = create_vlen_dset(grp, 'vlen_flat', compression=compression)
            frames_dset = grp.create_dataset('frames', shape=(0,), maxshape=(None,), dtype=np.int32,
                                             chunks=(256,), compression=compression, **(comp_opts if compression == 'gzip' else {}))
            labels_dset = grp.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype('utf-8'),
                                             chunks=(256,), compression=compression, **(comp_opts if compression == 'gzip' else {}))
            paths_dset = grp.create_dataset('paths', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype('utf-8'),
                                            chunks=(256,), compression=compression, **(comp_opts if compression == 'gzip' else {}))

        # process in parallel (CPU-heavy) and write in main process in batches
        batch_features = []
        batch_labels = []
        batch_paths = []
        batch_frames = []  # only for variable mode

        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {ex.submit(_process_file_worker, wi): wi[0] for wi in worker_inputs}
            if 'tqdm' in globals():
                pbar = tqdm(total=len(futures), desc='Processing files')
            else:
                pbar = None

            for fut in as_completed(futures):
                src_path = futures[fut]
                try:
                    ok, res = fut.result()
                except Exception as e:
                    ok = False
                    res = {'original_path': str(src_path), 'status': 'error_worker', 'error': str(e) + '\n' + traceback.format_exc()}

                if not ok:
                    errors.append(res)
                    manifest_rows.append({
                        'original_path': res.get('original_path', str(src_path)),
                        'feature_path': '',
                        'sr': args.sr,
                        'duration_sec': '',
                        'n_mels': args.n_mels,
                        'n_fft': args.n_fft,
                        'hop_length': args.hop_length,
                        'label': '',
                        'status': res.get('status', 'error'),
                        'error': res.get('error', '')
                    })
                else:
                    feat = res['feature']  # shape (n_mels, frames)
                    frames = res['frames']
                    meta = res['meta']
                    feature_relpath = f'h5://{out_h5.name}::features'  # conceptual pointer
                    manifest_rows.append({
                        'original_path': meta['original_path'],
                        'feature_path': str(out_h5),
                        'sr': meta['sr'],
                        'duration_sec': meta['duration_sec'],
                        'n_mels': meta['n_mels'],
                        'n_fft': meta['n_fft'],
                        'hop_length': meta['hop_length'],
                        'label': meta['label'],
                        'status': meta['status'],
                        'error': ''
                    })

                    if fixed_mode:
                        # validate frames equal
                        if frames != fixed_frames:
                            # if mismatch (rare due to edge effects), either pad/truncate along frame axis
                            if frames < fixed_frames:
                                pad_width = fixed_frames - frames
                                feat = np.pad(feat, ((0, 0), (0, pad_width)), mode='constant', constant_values=(np.min(feat)))
                            else:
                                feat = feat[:, :fixed_frames]
                        batch_features.append(feat)
                        batch_labels.append(meta['label'])
                        batch_paths.append(meta['original_path'])
                    else:
                        flat = feat.flatten()
                        batch_features.append(flat)
                        batch_frames.append(frames)
                        batch_labels.append(meta['label'])
                        batch_paths.append(meta['original_path'])

                # flush batch if large
                if len(batch_features) >= args.batch_write:
                    if fixed_mode:
                        append_fixed_dset(features_dset, batch_features)
                        # append labels & paths
                        cur = labels_dset.shape[0]
                        labels_dset.resize(cur + len(batch_labels), axis=0)
                        labels_dset[cur:cur + len(batch_labels)] = batch_labels
                        paths_dset.resize(cur + len(batch_paths), axis=0)
                        paths_dset[cur:cur + len(batch_paths)] = batch_paths
                    else:
                        append_vlen_dset(features_dset, batch_features)
                        cur = frames_dset.shape[0]
                        frames_dset.resize(cur + len(batch_frames), axis=0)
                        frames_dset[cur:cur + len(batch_frames)] = batch_frames
                        cur2 = labels_dset.shape[0]
                        labels_dset.resize(cur2 + len(batch_labels), axis=0)
                        labels_dset[cur2:cur2 + len(batch_labels)] = batch_labels
                        paths_dset.resize(cur2 + len(batch_paths), axis=0)
                        paths_dset[cur2:cur2 + len(batch_paths)] = batch_paths

                    batch_features.clear()
                    batch_labels.clear()
                    batch_paths.clear()
                    batch_frames.clear()

                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()

        # flush remaining batches
        if len(batch_features) > 0:
            if fixed_mode:
                append_fixed_dset(features_dset, batch_features)
                cur = labels_dset.shape[0]
                labels_dset.resize(cur + len(batch_labels), axis=0)
                labels_dset[cur:cur + len(batch_labels)] = batch_labels
                paths_dset.resize(cur + len(batch_paths), axis=0)
                paths_dset[cur:cur + len(batch_paths)] = batch_paths
            else:
                append_vlen_dset(features_dset, batch_features)
                cur = frames_dset.shape[0]
                frames_dset.resize(cur + len(batch_frames), axis=0)
                frames_dset[cur:cur + len(batch_frames)] = batch_frames
                cur2 = labels_dset.shape[0]
                labels_dset.resize(cur2 + len(batch_labels), axis=0)
                labels_dset[cur2:cur2 + len(batch_labels)] = batch_labels
                paths_dset.resize(cur2 + len(batch_paths), axis=0)
                paths_dset[cur2:cur2 + len(batch_paths)] = batch_paths

        # final metadata
        total_written = labels_dset.shape[0]
        h5f.attrs['num_samples'] = total_written
        h5f.attrs['errors'] = len(errors)

    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_root / args.manifest_name
    manifest_df.to_csv(manifest_path.as_posix(), index=False)

    print(f'Done. Written {total_written} samples to {out_h5}. Manifest: {manifest_path}. Errors: {len(errors)}')
    if len(errors) > 0:
        print('Some errors (up to 10):')
        for e in errors[:10]:
            print('-', e.get('original_path'), '|', e.get('status'), '|', e.get('error'))


if __name__ == '__main__':
    main()

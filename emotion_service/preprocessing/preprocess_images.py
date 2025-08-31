#!/usr/bin/env python3
"""
Preprocess images from AffectNet-like folder structure and save into an HDF5 file.

Expected source structure (default):
    data/images/{train|val|test}/{class_id}/imageXXX.jpg

Output:
    single HDF5 file with groups: /train, /val, /test (if present)
    each group contains datasets: images (float32, N,H,W,C), labels (int32), paths (variable-length utf-8)

Features:
 - optional face cropping using OpenCV Haar cascade (largest detected face)
 - center-crop to square + resize (default 224x224)
 - multi-threaded processing; main thread writes to HDF5 as results arrive (memory-efficient)
 - chunking + gzip compression
"""

from pathlib import Path
import argparse
import os
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import traceback
import numpy as np
import h5py
from collections import defaultdict
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# optional OpenCV face detection
USE_OPENCV = True
if USE_OPENCV:
    try:
        import cv2
        OPENCV_AVAILABLE = True
    except Exception:
        OPENCV_AVAILABLE = False
else:
    OPENCV_AVAILABLE = False


def detect_largest_face_bbox_opencv(pil_image):
    """Return bounding box (left, top, right, bottom) for largest face or None"""
    if not OPENCV_AVAILABLE:
        return None
    try:
        # PIL to BGR OpenCV
        im_arr = np.array(pil_image.convert('RGB'))[:, :, ::-1]
        gray = cv2.cvtColor(im_arr, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        largest = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = largest
        return (int(x), int(y), int(x + w), int(y + h))
    except Exception:
        return None


def center_crop_square(pil_img: Image.Image) -> Image.Image:
    """Center-crop the PIL image to a square (shorter side)."""
    short = min(pil_img.size)
    return ImageOps.fit(pil_img, (short, short), method=Image.LANCZOS, centering=(0.5, 0.5))


def preprocess_image_to_array(src_path: Path, size: int, face_crop: bool, grayscale: bool):
    """Load, optionally face-crop, center-crop, resize and return numpy array (H,W,C or H,W,1) scaled to [0,1]."""
    try:
        with Image.open(src_path) as im:
            if grayscale:
                im = im.convert('L')
            else:
                im = im.convert('RGB')

            if face_crop and OPENCV_AVAILABLE:
                try:
                    bbox = detect_largest_face_bbox_opencv(im)
                    if bbox is not None:
                        im = im.crop(bbox)
                except Exception:
                    # ignore face crop errors and fallback to center crop
                    pass

            im = center_crop_square(im)
            im = im.resize((size, size), Image.LANCZOS)

            arr = np.asarray(im, dtype=np.float32) / 255.0  # normalize to [0,1]
            # Ensure channel dimension format H,W,C (if grayscale, make channel dim)
            if grayscale:
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
            else:
                # PIL -> numpy already H,W,3 for RGB
                if arr.ndim == 2:
                    # weird single channel, convert to 3-channel
                    arr = np.stack([arr] * 3, axis=-1)
            return True, arr, None
    except Exception as e:
        tb = traceback.format_exc()
        return False, None, f"{e}\n{tb}"


def gather_image_files(src_root: Path, exts=None):
    """
    Return list of tuples: (abs_path, split, class_name, rel_path)
    split is one of 'train','val','test' if detected in the first path component, else 'all'
    """
    if exts is None:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = []
    for root, _dirs, fnames in os.walk(src_root):
        for f in fnames:
            if Path(f).suffix.lower() in exts:
                p = Path(root) / f
                rel = p.relative_to(src_root)
                parts = rel.parts
                if len(parts) >= 3 and parts[0].lower() in ('train', 'val', 'test'):
                    split = parts[0].lower()
                    class_name = parts[1]
                    class_rel = Path(*parts[1:])  # e.g., class/file.jpg
                elif len(parts) >= 2:
                    # treat top dir as class, split = all
                    split = 'all'
                    class_name = parts[0]
                    class_rel = Path(*parts)
                else:
                    split = 'all'
                    class_name = 'unknown'
                    class_rel = rel
                files.append((p, split, class_name, class_rel))
    # sort deterministic
    files.sort(key=lambda x: (x[1], str(x[2]), str(x[3])))
    return files


def create_or_get_group_datasets(h5f, group_name, img_shape, dtype=np.float32, chunk_size=128, compression='gzip', comp_level=4):
    """
    Ensure group exists and datasets images, labels, paths exist with extendable first dim.
    Returns (images_ds, labels_ds, paths_ds, current_len)
    """
    grp = h5f.require_group(group_name)
    # determine dataset names
    img_ds_name = 'images'
    lbl_ds_name = 'labels'
    paths_ds_name = 'paths'

    H, W, C = img_shape
    if img_ds_name not in grp:
        maxshape = (None, H, W, C)
        # choose chunk shape sensibly
        chunk = (min(chunk_size, 16), H, W, C)
        grp.create_dataset(img_ds_name, shape=(0, H, W, C), maxshape=maxshape,
                           dtype=dtype, chunks=chunk, compression=compression, compression_opts=comp_level)
    if lbl_ds_name not in grp:
        grp.create_dataset(lbl_ds_name, shape=(0,), maxshape=(None,), dtype=np.int32,
                           chunks=(min(chunk_size, 1024),), compression=compression, compression_opts=comp_level)
    if paths_ds_name not in grp:
        str_dt = h5py.string_dtype(encoding='utf-8')
        grp.create_dataset(paths_ds_name, shape=(0,), maxshape=(None,), dtype=str_dt,
                           chunks=(min(chunk_size, 1024),), compression=compression, compression_opts=comp_level)
    images_ds = grp[img_ds_name]
    labels_ds = grp[lbl_ds_name]
    paths_ds = grp[paths_ds_name]
    current_len = images_ds.shape[0]
    return images_ds, labels_ds, paths_ds, current_len


def append_to_datasets(images_ds, labels_ds, paths_ds, arr, label, relpath):
    """Append single sample to the three datasets (resize then write)."""
    n = images_ds.shape[0]
    images_ds.resize(n + 1, axis=0)
    images_ds[n, ...] = arr
    labels_ds.resize(n + 1, axis=0)
    labels_ds[n] = int(label)
    paths_ds.resize(n + 1, axis=0)
    paths_ds[n] = str(relpath)
    return n + 1


def parse_args():
    p = argparse.ArgumentParser(description='Preprocess images and save to HDF5')
    p.add_argument('--src', type=str, default=str(Path(__file__).parents[0] / 'data' / 'images'),
                   help='source images root')
    p.add_argument('--out', type=str, default=str(Path(__file__).parents[0] / 'preprocessed_images.h5'),
                   help='output HDF5 file path')
    p.add_argument('--size', type=int, default=224, help='output size (square), default 224')
    p.add_argument('--face-crop', action='store_true', help='attempt to detect and crop faces before resizing (uses OpenCV)')
    p.add_argument('--grayscale', action='store_true', help='convert output to grayscale (1 channel)')
    p.add_argument('--workers', type=int, default=8, help='number of worker threads (default 8)')
    p.add_argument('--chunk', type=int, default=256, help='HDF5 chunk size (number of samples per chunk)')
    p.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'lzf', 'none'], help='dataset compression')
    p.add_argument('--compression-level', type=int, default=4, help='gzip compression level (1-9)')
    p.add_argument('--no-progress', action='store_true', help='disable tqdm progress bar')
    p.add_argument('--overwrite', action='store_true', help='overwrite existing HDF5 file')
    return p.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not src_root.exists():
        print(f'ERROR: source root does not exist: {src_root}')
        return

    if out_path.exists():
        if args.overwrite:
            out_path.unlink()
        else:
            print(f'ERROR: output file exists ({out_path}). Use --overwrite to replace.')
            return

    print(f"Source: {src_root}")
    print(f"Output HDF5: {out_path}")
    print(f"Size: {args.size}x{args.size}, face_crop={args.face_crop}, grayscale={args.grayscale}, workers={args.workers}")

    files = gather_image_files(src_root)
    if len(files) == 0:
        print('No image files found under source root. Exiting.')
        return

    # Build class mapping from class_name -> int
    classes = sorted({c for (_, _split, c, _) in files})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Detected {len(classes)} classes.")

    # determine channels
    channels = 1 if args.grayscale else 3
    img_shape = (args.size, args.size, channels)

    use_tqdm = (tqdm is not None) and (not args.no_progress)
    pbar = tqdm(total=len(files), desc='processing') if use_tqdm else None

    # open HDF5 for writing
    compression = None if args.compression == 'none' else args.compression
    with h5py.File(out_path, 'w') as h5f:
        # store metadata attributes
        h5f.attrs['source_root'] = str(src_root)
        h5f.attrs['size'] = args.size
        h5f.attrs['grayscale'] = bool(args.grayscale)
        h5f.attrs['face_crop'] = bool(args.face_crop)
        h5f.attrs['channels'] = channels
        h5f.attrs['total_images_found'] = len(files)
        # store class mapping as group attribute (string)
        mapping_grp = h5f.create_group('mapping')
        mapping_grp.attrs['num_classes'] = len(classes)
        for name, idx in class_to_idx.items():
            mapping_grp.attrs[name] = int(idx)

        # We will create datasets lazily per split when first sample arrives
        datasets_info = {}  # split -> (images_ds, labels_ds, paths_ds)

        fn = partial(preprocess_image_to_array, size=args.size, face_crop=args.face_crop, grayscale=args.grayscale)

        # Use ThreadPoolExecutor for CPU-bound IO + small CPU ops; could be ProcessPool for heavy CPU ops
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {ex.submit(fn, p): (p, split, class_name, rel) for (p, split, class_name, rel) in files}
            processed = 0
            errors = []
            for fut in as_completed(futures):
                src_path, split, class_name, rel = futures[fut]
                try:
                    ok, arr, err = fut.result()
                except Exception as e:
                    ok = False
                    arr = None
                    err = f"uncaught error: {e}\n{traceback.format_exc()}"

                if not ok:
                    errors.append((str(src_path), err))
                else:
                    # ensure correct shape: (H,W,C)
                    if arr.shape != img_shape:
                        # handle channel mismatches (e.g., grayscale returned as H,W,1)
                        if arr.ndim == 2:
                            arr = arr[:, :, np.newaxis]
                        if arr.shape[2] != channels:
                            # convert channels if necessary (rare)
                            if channels == 1:
                                # convert to luminance
                                arr = np.mean(arr, axis=2, keepdims=True)
                            elif channels == 3 and arr.shape[2] == 1:
                                arr = np.repeat(arr, 3, axis=2)
                            arr = arr.astype(np.float32)
                    grp_name = split
                    if grp_name not in datasets_info:
                        images_ds, labels_ds, paths_ds, cur = create_or_get_group_datasets(
                            h5f, grp_name, img_shape, dtype=np.float32, chunk_size=args.chunk,
                            compression=compression, comp_level=args.compression_level)
                        datasets_info[grp_name] = (images_ds, labels_ds, paths_ds)
                    else:
                        images_ds, labels_ds, paths_ds = datasets_info[grp_name]
                    # Append (main thread)
                    append_to_datasets(images_ds, labels_ds, paths_ds, arr, class_to_idx[class_name], rel)
                processed += 1
                if use_tqdm:
                    pbar.update(1)
            if use_tqdm:
                pbar.close()

        # After processing, write summary attributes
        total_written = sum(h5f[grp].attrs.get('written', h5f[grp]['images'].shape[0]) if 'images' in h5f[grp] else 0 for grp in h5f.keys() if isinstance(h5f[grp], h5py.Group))
        h5f.attrs['processed_files'] = processed
        h5f.attrs['errors'] = len(errors)

        # Save per-group counts as attributes
        for grp in h5f:
            if isinstance(h5f[grp], h5py.Group) and 'images' in h5f[grp]:
                h5f[grp].attrs['num_samples'] = h5f[grp]['images'].shape[0]

    # print summary
    print(f"\nDone. Processed: {processed}, Errors: {len(errors)}")
    if len(errors) > 0:
        print("Some failures (showing up to 10):")
        for s, e in errors[:10]:
            print("-", s, "|", e)


if __name__ == '__main__':
    main()

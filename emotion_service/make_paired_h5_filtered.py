import h5py, os, random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def basename(x):
    if isinstance(x, bytes):
        x = x.decode()
    return os.path.basename(str(x))

# Mapping from numeric image labels to emotion names
mapping = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']

# Map audio dataset label names to image label names
label_map = {
    'anger': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad'
}

def pair_and_write_embeddings(imf, audf, out_prefix, seed=1234):
    rng = random.Random(seed)
    allowed = list(label_map.values())  # ['angry','disgust','fear','happy','neutral','sad']

    # Load audio embeddings
    aud_feats = audf['all/embeddings']
    aud_labels_raw = audf['all/labels'][:]
    aud_paths = audf['all/paths'][:]

    # Decode and map audio labels
    aud_labels = []
    for x in aud_labels_raw:
        if isinstance(x, bytes):
            x = x.decode().strip().lower()
        else:
            x = str(x).strip().lower()
        if x in label_map:
            x = label_map[x]
        aud_labels.append(x)

    # Group audio by emotion
    aud_by_class = defaultdict(list)
    for i, lab in enumerate(aud_labels):
        if lab in allowed:
            aud_by_class[lab].append(i)

    print("\nAudio samples per allowed class:")
    for cls in allowed:
        print(f"  {cls}: {len(aud_by_class[cls])}")

    total_pairs_all = 0

    # Process each split
    for split in ['train','val','test']:
        im_emb = imf[f"{split}/embeddings"]
        im_labels_raw = imf[f"{split}/labels"][:]
        im_paths = imf[f"{split}/paths"][:]
        im_label_names = [mapping[int(l)] for l in im_labels_raw]

        print(f"\nImage samples in split '{split}':")
        for cls in allowed:
            count = sum(1 for lab in im_label_names if lab == cls)
            print(f"  {cls}: {count}")

        assigned_pairs = []
        for cls in allowed:
            img_idxs = [i for i, lab in enumerate(im_label_names) if lab == cls]
            aud_pool = aud_by_class.get(cls, [])
            if not img_idxs or not aud_pool:
                continue

            rng.shuffle(aud_pool)
            while len(aud_pool) < len(img_idxs):
                aud_pool.extend(aud_pool)

            for imi, ai in zip(img_idxs, aud_pool[:len(img_idxs)]):
                assigned_pairs.append((imi, ai))

        if not assigned_pairs:
            print(f"⚠️ No pairs created for split '{split}'.")
            continue

        N = len(assigned_pairs)
        total_pairs_all += N
        out_path = f"{out_prefix}_{split}_paired_embeddings.h5"

        with h5py.File(out_path, 'w') as outf:
            outf.create_dataset('image_embeddings', shape=(N, im_emb.shape[1]), dtype=im_emb.dtype)
            outf.create_dataset('audio_embeddings', shape=(N, aud_feats.shape[1]), dtype=aud_feats.dtype)
            dt = h5py.string_dtype('utf-8')
            outf.create_dataset('labels', shape=(N,), dtype=dt)
            outf.create_dataset('paths', shape=(N,), dtype=dt)

            for i, (imi, ai) in enumerate(tqdm(assigned_pairs, desc=f"{split} pairs", unit="pair")):
                outf['image_embeddings'][i] = im_emb[imi]
                outf['audio_embeddings'][i] = aud_feats[ai]
                outf['labels'][i] = im_label_names[imi]
                outf['paths'][i] = basename(im_paths[imi])

        print(f"✅ Wrote {N} pairs to: {out_path}")

    print(f"\n✅ Finished pairing. Total pairs created: {total_pairs_all}")

if __name__ == '__main__':
    image_h5 = "embeddings_images.h5"
    audio_h5 = "embeddings_audio.h5"
    out_prefix = "saved_paired"
    seed = 1234

    with h5py.File(image_h5, 'r') as imf, h5py.File(audio_h5, 'r') as audf:
        pair_and_write_embeddings(imf, audf, out_prefix, seed=seed)

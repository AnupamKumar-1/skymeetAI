import h5py

def inspect(path):
    print(f"\nInspecting {path}")
    with h5py.File(path, "r") as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"[DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"[GROUP]   {name}")
        f.visititems(print_structure)

inspect("embeddings_audio.h5")
inspect("embeddings_images.h5")

# import h5py

# audio_h5 = "embeddings_audio.h5"

# with h5py.File(audio_h5, 'r') as f:
#     labels = [str(x).strip().lower() for x in f['all/labels'][:]]
#     unique_labels = sorted(set(labels))
#     print("Unique audio labels:", unique_labels)
#     print("Counts per label:")
#     for lbl in unique_labels:
#         print(lbl, labels.count(lbl))

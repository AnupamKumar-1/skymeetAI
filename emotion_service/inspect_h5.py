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


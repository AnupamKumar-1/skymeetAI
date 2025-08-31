import h5py
import torch
import numpy as np
from tqdm import tqdm
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== IMAGE MODEL ======
image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
image_model.fc = torch.nn.Identity()  # remove classification head
image_model = image_model.to(DEVICE).eval()

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ====== AUDIO MODEL ======
class AudioMLP(torch.nn.Module):
    def __init__(self, input_dim, emb_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, emb_dim)
        )
    def forward(self, x):
        return self.net(x)

# Detect MFCC dimension
with h5py.File("./preprocessed_audio/preprocessed_audio.h5", "r") as af:
    sample_shape = af["features"]["fixed"][0].shape
    audio_input_dim = int(np.prod(sample_shape))

audio_model = AudioMLP(audio_input_dim).to(DEVICE).eval()


# ====== IMAGE EMBEDDING EXTRACTION ======
def extract_image_embeddings(input_h5, output_h5):
    with h5py.File(input_h5, "r") as fin, h5py.File(output_h5, "w") as fout:
        for split in ["train", "val", "test"]:
            if split not in fin:
                print(f"Skipping {split} (not found in images)")
                continue

            imgs = fin[split]["images"]
            labels = fin[split]["labels"][:]
            paths = fin[split]["paths"][:]

            emb_list = []
            for i in tqdm(range(len(imgs)), desc=f"Image {split}"):
                img = imgs[i]
                img = img_transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = image_model(img).cpu().numpy()
                emb_list.append(emb)

            emb_arr = np.vstack(emb_list)

            grp = fout.create_group(split)
            grp.create_dataset("embeddings", data=emb_arr)
            grp.create_dataset("labels", data=labels)
            grp.create_dataset("paths", data=paths)
        print(f"Image embeddings saved to {output_h5}")


# ====== AUDIO EMBEDDING EXTRACTION ======
def extract_audio_embeddings(input_h5, output_h5):
    with h5py.File(input_h5, "r") as fin, h5py.File(output_h5, "w") as fout:
        fixed = fin["features"]["fixed"]
        labels = fin["features"]["labels"][:]
        paths = fin["features"]["paths"][:]

        emb_list = []
        for i in tqdm(range(len(fixed)), desc="Audio all"):
            x = torch.tensor(fixed[i].flatten(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = audio_model(x).cpu().numpy()
            emb_list.append(emb)

        emb_arr = np.vstack(emb_list)

        grp = fout.create_group("all")
        grp.create_dataset("embeddings", data=emb_arr)
        grp.create_dataset("labels", data=labels)
        grp.create_dataset("paths", data=paths)
        print(f"Audio embeddings saved to {output_h5}")


# ====== RUN ======
extract_image_embeddings(
    "./preprocessed_images/preprocessed_images.h5",
    "./embeddings_images.h5"
)

extract_audio_embeddings(
    "./preprocessed_audio/preprocessed_audio.h5",
    "./embeddings_audio.h5"
)

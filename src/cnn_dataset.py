"""Dataset PyTorch para as imagens de mamografia recortadas (CBIS-DDSM)."""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 224
IMAGENET_LIKE_MEAN = 0.5
IMAGENET_LIKE_STD = 0.5


class MammographyDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int = IMAGE_SIZE, augment: bool = False):
        self.paths = manifest["full_path"].tolist()
        self.labels = manifest["label"].tolist()
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path) -> np.ndarray:
        img = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int):
        image = self._load_image(self.paths[idx])
        image = (image - IMAGENET_LIKE_MEAN) / IMAGENET_LIKE_STD
        tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                tensor = torch.flip(tensor, dims=[2])  # flip horizontal
            if torch.rand(1).item() < 0.5:
                tensor = torch.flip(tensor, dims=[1])  # flip vertical

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MammographyDatasetRGB(Dataset):
    """Variante para transfer learning: replica o canal de tons de cinza em 3 canais
    e normaliza com as estatísticas do ImageNet, como esperado por CNNs pré-treinadas
    (ex.: ResNet18)."""

    def __init__(self, manifest: pd.DataFrame, image_size: int = IMAGE_SIZE, augment: bool = False):
        self.paths = manifest["full_path"].tolist()
        self.labels = manifest["label"].tolist()
        self.image_size = image_size
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path) -> np.ndarray:
        img = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int):
        image = self._load_image(self.paths[idx])
        tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if torch.rand(1).item() < 0.5:
                tensor = torch.flip(tensor, dims=[1])

        tensor = (tensor - self.mean) / self.std
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label

"""Download EuroSAT (RGB), make a fixed stratified split and build PyTorch data loaders.

EuroSAT: 27,000 Sentinel-2 satellite image patches (64x64 px, 10 m per pixel) in 10 land-use classes.
"""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eurosat"
URL = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip"
SPLIT_FILE = ROOT / "data" / "splits.csv"
SEED = 42

# ImageNet statistics, because the pretrained backbone was trained with them
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def download():
    zip_path = DATA / "EuroSAT.zip"
    DATA.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print("downloading EuroSAT (~90 MB)")
        urllib.request.urlretrieve(URL, zip_path)
    if not (DATA / "2750").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA)
    return DATA / "2750"


def make_splits(image_dir: Path) -> pd.DataFrame:
    """70 / 15 / 15 stratified split, saved to CSV so every model sees exactly the same test images."""
    if SPLIT_FILE.exists():
        return pd.read_csv(SPLIT_FILE)
    rows = [{"path": str(p.relative_to(image_dir)), "label": p.parent.name}
            for p in sorted(image_dir.glob("*/*.jpg"))]
    df = pd.DataFrame(rows)
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    df.to_csv(SPLIT_FILE, index=False)
    return df


def class_names(df: pd.DataFrame) -> list:
    return sorted(df["label"].unique())


def build_transforms(train: bool, size: int):
    # Satellite images have no "up": flips and 90-degree rotations are all valid views of the same place
    steps = [transforms.Resize(size)]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
                  transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class EuroSAT(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, classes: list, transform):
        self.paths = [image_dir / p for p in df["path"]]
        self.targets = [classes.index(c) for c in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.targets[i]


def subsample(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Stratified fraction of the training set, used for the 'how many labels do we need' experiment."""
    if fraction >= 1:
        return df
    part, _ = train_test_split(df, train_size=fraction, stratify=df["label"], random_state=SEED)
    return part


def get_loaders(size: int = 128, batch_size: int = 64, train_fraction: float = 1.0):
    image_dir = download()
    df = make_splits(image_dir)
    classes = class_names(df)
    train_df = subsample(df[df.split == "train"], train_fraction)
    loaders = {}
    for name, part in [("train", train_df), ("val", df[df.split == "val"]), ("test", df[df.split == "test"])]:
        ds = EuroSAT(part, image_dir, classes, build_transforms(name == "train", size))
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=name == "train", num_workers=2,
                                   persistent_workers=True, generator=torch.Generator().manual_seed(SEED))
    return loaders, classes

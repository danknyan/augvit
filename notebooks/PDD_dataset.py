from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
import torch
import pandas as pd
import numpy as np

class CropDiseaseDataset(Dataset):

    def __init__(self, root: Path, img_size: int):
        
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            #v2.ToDtype(torch.float32, scale=True),
        ])

        self.samples: list = []
        self.class_to_idx: dict = {}
        self.class_names = []

        for idx, class_dir in enumerate(sorted(root.iterdir())):
            if not class_dir.is_dir():
                continue
            self.class_to_idx[class_dir.name] = idx
            self.class_names.append(class_dir.name)
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, idx))

        paths, labels = zip(*self.samples)
        self.paths  = np.array([str(p) for p in paths])   # numpy array, not list
        self.labels = np.array(labels, dtype=np.int64)
        self.cache = {}
        del self.samples 

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx):
        if idx not in self.cache:
            img = read_image(self.paths[idx], mode=ImageReadMode.RGB)
            self.cache[idx] = self.transform(img)
        return self.cache[idx], int(self.labels[idx])
    
class OGDiseaseDataset(Dataset):
    def __init__(self, root, img_size: int):
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            #v2.ToDtype(torch.float32, scale=True),
        ])

        df = pd.read_csv(root+"/_classes.csv")
        df.columns = df.columns.str.strip()
        df["filename"] = df["filename"].str.strip()
        # drop classes with less than 10 instances 
        temp_labels = df[[c for c in df.columns if c != "filename"]].values.argmax(axis=1)
        unique_labels, counts = np.unique(temp_labels, return_counts=True)
        valid_classes = unique_labels[counts >= 10]
        mask = np.isin(temp_labels, valid_classes)
        # clean up DF
        df = df[mask].reset_index(drop=True)
        df = df.loc[:, (df != 0).any(axis=0)]

        # for loaders
        class_names = [c for c in df.columns if c != "filename"]
        labels = df[class_names].values.argmax(axis=1)

        self.samples: list = [(Path(root) / Path(fname),int(label)) for fname,label in zip( df["filename"],labels)]
        self.class_to_idx: dict = {label:ind for ind,label in enumerate(np.unique(class_names))}
        self.class_names = class_names
        paths, labels = zip(*self.samples)
        self.paths  = np.array([str(p) for p in paths])   # numpy array, not list
        self.labels = np.array(labels, dtype=np.int64)
        self.cache = {}
        del self.samples 

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img = read_image(self.paths[idx], mode=ImageReadMode.RGB)
            self.cache[idx] = self.transform(img)
        return self.cache[idx], int(self.labels[idx])
    
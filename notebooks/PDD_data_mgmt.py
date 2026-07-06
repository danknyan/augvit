from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
import torch
import pandas as pd
import numpy as np


# main dataset, loads Tensors of images, not images
class CropDiseaseDataset(Dataset):

    def __init__(self, root: str,):
        root = Path(root)
        self.samples: list = []
        self.class_to_idx: dict = {}
        self.class_names = []

        for idx, class_dir in enumerate(sorted(root.iterdir())):
            if not class_dir.is_dir():
                continue
            self.class_to_idx[class_dir.name] = idx
            self.class_names.append(class_dir.name)
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() == ".pt":     
                    self.samples.append((img_path, idx))

        paths, labels = zip(*self.samples)
        self.paths  = np.array([str(p) for p in paths])
        self.labels = np.array(labels, dtype=np.int64)
        del self.samples

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        tensor = torch.load(self.paths[idx], weights_only=True)
        return tensor, int(self.labels[idx])

# loads images into tensors
class OGDiseaseDataset(Dataset):
    def __init__(self, root, img_size: int):

        # baseline transforms
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.ToDtype(torch.float32, scale=True),
        ])
        # gather class data
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


# datasets
# preload the dataset into memory to run save time on sequential training
class PreBatchedDataset:
    def __init__(self, dataset, batch_size, shuffle=True):
        all_data   = torch.stack([dataset[i][0] for i in range(len(dataset))])
        all_labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])

        if shuffle:
            idx = torch.randperm(len(all_data))
            all_data, all_labels = all_data[idx], all_labels[idx]

        n = (len(all_data) // batch_size) * batch_size
        self.data   = all_data[:n]    # regular RAM — no pin_memory()
        self.labels = all_labels[:n]
        self.batch_size = batch_size
        self.n_batches  = n // batch_size

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        # bounds check
        if idx >= self.n_batches or idx < 0:
            raise IndexError("Batch index out of range")
        s = idx * self.batch_size
        return self.data[s:s+self.batch_size], self.labels[s:s+self.batch_size]

    def shuffle(self):
        idx = torch.randperm(len(self.data))
        self.data   = self.data[idx]
        self.labels = self.labels[idx]

# prefetcher to load to GPU and manage memory
class CUDAStreamPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._next_data   = None
        self._next_labels = None
        self._preload_iter = None

    def __iter__(self):
        self._next_data   = None
        self._next_labels = None
        torch.cuda.empty_cache()

        self._preload_iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self._next_data, self._next_labels = next(self._preload_iter)
        except StopIteration:
            self._next_data = None
            return
        with torch.cuda.stream(self.stream):
            self._next_data   = self._next_data.to(self.device, non_blocking=True)
            self._next_labels = self._next_labels.to(self.device, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data, labels = self._next_data, self._next_labels
        if data is None:
            raise StopIteration
        self._preload()
        return data, labels

    def __len__(self):
        return len(self.loader)
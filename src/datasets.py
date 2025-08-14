import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X) if not isinstance(X, torch.Tensor) else X
        self.y = torch.from_numpy(y) if not isinstance(y, torch.Tensor) else y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]            # (T, C)
        return x, self.y[idx]

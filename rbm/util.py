import torch
from torch.utils.data import Dataset, DataLoader

class RBMDataset(Dataset):
    def __init__(self, X_continuous: torch.Tensor | None, 
                 X_binary: torch.Tensor | None):
        self.X_continuous = X_continuous,
        self.X_binary = X_binary
    def __len__(self):
        return len(self.X_continuous)
    def __getitem__(self, index):
        sample = {'X_continuous': None, 'X_binary': None}
        if self.X_continuous is not None:
            sample['X_continuous'] = self.X_continuous[index]
        if self.X_binary is not None:
            sample['X_binary'] = self.X_binary[index]
        return sample

def build_dataloader(X_continuous: torch.Tensor | None, 
                     X_binary: torch.Tensor | None,
                     batch_size: int = 1):
    return DataLoader(dataset=RBMDataset(X_continuous, X_binary), 
                            batch_size=batch_size, shuffle=True)


from torch.utils.data import Dataset
import torch

class AISDataset(Dataset):
    def __init__(self, data, labels, masks, padding_masks):
        self.data = data
        self.labels = labels
        self.masks = masks
        self.padding_masks = padding_masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = {
            "observed_data": torch.tensor(self.data[idx], dtype=torch.float32), # [maxlen, n]
            "observed_labels": torch.tensor(self.labels[idx], dtype=torch.float32),# [maxlen, n]
            "masks": torch.tensor(self.masks[idx], dtype=torch.bool), # [maxlen, n]
            "padding_masks": torch.tensor(self.padding_masks[idx], dtype=torch.bool), # [maxlen, n]
        }
        return s
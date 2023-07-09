from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import cleantext


class ToxicDataset(Dataset):
    def __init__(self, csv_path, is_training):
        self.is_training = is_training
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row.comment_text
        label = row.toxic
        text = cleantext.clean(text)
        return text, torch.tensor(label).unsqueeze(0).float()

    
import torch
import pandas as pd
import numpy as np
# from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader

from data_loader.fast_data_loader import FastTensorDataLoader


class TextDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()

        if isinstance(file_path, pd.DataFrame):
            self.data = file_path
        else:
            self.data = pd.read_parquet(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        x = torch.tensor(row['text_emb'])
        y = torch.tensor(row['label']).float()

        return x, y


class TextDatasetWithSource(TextDataset):
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        x = torch.tensor(row['text_emb'])
        y = torch.tensor(row['label']).float()
        y2 = torch.tensor(row['src']).float()

        return x, y, y2


class ModelRatingsDataset(TextDataset):
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        x = torch.tensor(row['text_emb'])
        y = torch.LongTensor([row['rating'].item()])

        return x, y


class TextDatasetWithSourceDF(TextDatasetWithSource):
    """
    Load a TextDatasetWithSource loader from a DataFrame
    """
    def __init__(self, data_frame):
        self.data = data_frame


class DomainDataset(Dataset):
    def __init__(self, dataset_1: Dataset, dataset_2: Dataset, dataset_3: Dataset, dataset_4: Dataset=None):  # , dataset_4: Dataset
        super().__init__()

        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.dataset_3 = dataset_3
        if dataset_4 is not None:
            self.dataset_4 = dataset_4
        else:
            self.dataset_4 =None
        self._len_1 = len(self.dataset_1)
        self._len_2 = len(self.dataset_2)
        self._len_3 = len(self.dataset_3)
        self._len_4 = len(self.dataset_4) if self.dataset_4 is not None else 0

    def __len__(self):
        return self._len_1 + self._len_2 + self._len_3  + self._len_4

    def __getitem__(self, idx):
        if idx < self._len_1:
            label = 0
            row = self.dataset_1[idx]
        elif idx >= self._len_1 and idx < self._len_1 + self._len_2:
            label = 1
            row = self.dataset_2[idx - self._len_1]
        elif idx >= self._len_1 + self._len_2 and idx < self._len_1 + self._len_2 + self._len_3:
            label = 2
            row = self.dataset_3[idx - self._len_1 - self._len_2]
        else:
            label = 3
            row = self.dataset_4[idx - self._len_1 - self._len_2 - self._len_3]

        x = row[0]
        y = torch.FloatTensor([label])

        return x, y


class TextDataDataset(Dataset):
    def __init__(self, data):
        super().__init__()

        if hasattr(data, 'data'):
            self.data = data.data
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        x = torch.tensor(row['text_emb'])
        y = torch.tensor(row['label']).float()

        return x, y


class MultiDataset(Dataset):
    def __init__(self, dataset_1: Dataset, dataset_2: Dataset, dataset_3: Dataset, dataset_4: Dataset=None):
        super().__init__()

        self.dataset_1 = TextDataDataset(dataset_1)
        self.dataset_2 = TextDataDataset(dataset_2)

        self.dataset_3 = TextDataDataset(dataset_3)

        if dataset_4 is not None:
            self.dataset_4 = TextDataDataset(dataset_4)

        self._len_1 = len(self.dataset_1)
        self._len_2 = len(self.dataset_2)
        self._len_3 = len(self.dataset_3)
        self._len_4 = len(self.dataset_4) if dataset_4 is not None else 0

    def __len__(self):
        return self._len_1 + self._len_2 + self._len_3 + self._len_4

    def __getitem__(self, idx):
        if idx < self._len_1:
            return self.dataset_1[idx]
        elif idx >= self._len_1 and idx < self._len_1 + self._len_2:
            return self.dataset_2[idx - self._len_1]
        elif idx >= self._len_1 + self._len_2 and idx < self._len_1 + self._len_2 + self._len_3:
            return self.dataset_3[idx - self._len_1 - self._len_2]
        else:
            return self.dataset_4[idx - self._len_1 - self._len_2 - self._len_3]


class TextDatasetPairWrapper(Dataset):
    def __init__(self, dataset: TextDataset):
        super().__init__()
        self._data = dataset.data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data.iloc[idx]

        x = torch.tensor(row['text_emb'])
        y = torch.tensor(row['label']).float()

        target_label = 1 if np.random.rand() >= .5 else 0
        df = self._data
        pair_row = df[df['label'] == target_label].sample().iloc[0]

        x2 = torch.tensor(pair_row['text_emb'])
        y2 = torch.tensor(1 if pair_row['label'] == row['label'] else -1)

        return x, y, x2, y2


def create_fast_dataloader_from_dataset(dataset: Dataset, batch_size=32, shuffle=False):
    data = [dataset[i] for i in range(len(dataset))]
    dimensions = len(data[0])

    data = [torch.stack([x[i] for x in data]) for i in range(dimensions)]

    return FastTensorDataLoader(*data, batch_size=batch_size, shuffle=shuffle)

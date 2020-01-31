from torch.utils.data import Dataset
import torch
import numpy as np


class NTU_RGB_D_Dataset(Dataset):
    def __init__(self, data, label):
        super(Dataset, self).__init__()
        assert isinstance(data, np.ndarray)
        self.data = torch.from_numpy(data)
        self.label = torch.tensor(label[1])
        if torch.cuda.is_available():
            self.data = self.data.cuda()
            self.label = self.label.cuda()

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.label[index]

    def __len__(self):
        return len(self.label)

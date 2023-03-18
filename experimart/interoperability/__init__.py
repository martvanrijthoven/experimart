import torch.utils.data as data
import numpy as np
import glob
import os
import torch

class NPZDataLoader(data.Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.files = glob.glob(os.path.join(folder_path,'training','*.npz'))
    
    def __getitem__(self, index):
        data = np.load(self.files[index])
        return torch.from_numpy(data['x_batch']), torch.from_numpy(data['y_batch'])

    def __len__(self):
        return len(self.files)
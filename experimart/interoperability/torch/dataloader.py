import torch.utils.data as data
import numpy as np
import glob
import os

class SegmentationDataLoader(data.Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.data_files = glob.glob(os.path.join(folder_path,'*.npz'))

    def __getitem__(self, index):
        data_path =  self.data_files[index]
        data = np.load(data_path)
        return data['x_batch'], data['y_batch']

    def __len__(self):
        return len(self.data_files)
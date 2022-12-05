import torch
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, filePath, batch_size):
        self.half_batch_size = batch_size // 2
        self.batch_size = self.half_batch_size * 2
        self.df = self.get_dataframe(filePath)
        self.reset()
    
    @staticmethod
    def get_dataframe(filePath):
        df = pd.read_csv(filePath)
        df.drop([col for col in df.columns if 'Unnamed' in col], axis=1, inplace=True)
        return df
    
    def __len__(self):
        return self.df.shape[0]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self._zeros.shape[0] <= self.batch_size and \
                self._ones.shape[0] <= self.batch_size:
            raise StopIteration
        
        next_batch = self.get_next_batch()
        features = next_batch.drop(columns=['label'], axis=1).to_numpy()
        labels = next_batch.label.to_numpy()
        
        features_tensor = torch.tensor(features).to_sparse()
        labels_tensor = torch.tensor(labels, dtype=torch.float64)
        labels_tensor = labels_tensor.view(self.batch_size, 1)
        
        return features_tensor, labels_tensor
    
    def get_next_batch(self):
        batch_indeces_zeros = np.random.choice(self._zeros, self.half_batch_size, replace=False)
        self._zeros = np.setdiff1d(self._zeros, batch_indeces_zeros)

        batch_indeces_ones = np.random.choice(self._ones, self.half_batch_size, replace=False)
        self._ones = np.setdiff1d(self._ones, batch_indeces_ones)
        
        batch_indeces = np.union1d(batch_indeces_zeros, batch_indeces_ones)
        np.random.shuffle(batch_indeces)
        
        return self.df.iloc[batch_indeces]
    
    def reset(self):
        self._zeros = self.df[self.df.label == 0].index.to_numpy()
        self._ones = self.df[self.df.label == 1].index.to_numpy()
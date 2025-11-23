import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config
import ast

# 1. Define dataset
class InputMethodDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        df['input'] = df['input'].apply(ast.literal_eval)
        self.data = df.to_dict(orient='records')    # list of {'input': [...], 'target': ...}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['target'], dtype=torch.long)
        return input_tensor, target_tensor

# 2. Provide dataloader methods
def get_dataloader(train=True, shuffle=True):
    path = config.PROCESSED_DATA_DIR / ("train.csv" if train else "test.csv")
    dataset = InputMethodDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)
    

if __name__ == "__main__":
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)

    print(len(train_dataloader))
    print(len(test_dataloader))
    
    for input, target in train_dataloader:
        print(input.shape)
        print(target.shape)
        break
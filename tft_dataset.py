import pandas as pd
import numpy as np
import torch
import re

CHAMPIONS_PER_ROW = 12
COMBINATIONS_PER_ROW = 24
ITEMS_PER_CHAMPION = 3
MAX_NUM_COMBINATIONS = 10

NUM_CHAMPIONS = 52 + 1
NUM_ITEMS = 99 + 1
NUM_STARS = 3 + 1

class TftDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_path: str="C:/Projects/TFTPlaystylesAnalysis/training_data.csv"
    ):
        data = pd.read_csv(data_path)
        
        self.combination_cols = [col for col in data.columns if col.startswith("combination")]
        self.combination_data = data[self.combination_cols].astype(np.float32) / MAX_NUM_COMBINATIONS
        
        self.name_cols = [col for col in data.columns if re.match("character_.*_name", col)]
        self.name_data = data[self.name_cols]
        
        self.item_cols = [col for col in data.columns if re.match("character_.*_item_[123]", col)]
        self.item_data = data[self.item_cols]
        
        self.star_cols = [col for col in data.columns if re.match("character_.*_star", col)]
        self.star_data = data[self.star_cols]
    
    def __getitem__(self, index):
        return {
            "champions": self.name_data.iloc[index].to_numpy(),
            "champions_mask": self.name_data.iloc[index].to_numpy() == 0,
            "champion_items": self.item_data.iloc[index].to_numpy(),
            "champion_items_mask": self.item_data.iloc[index].to_numpy() == 0,
            "champion_star": self.star_data.iloc[index].to_numpy(),
            "champion_star_mask": self.star_data.iloc[index].to_numpy() == 0,
            "combinations": self.combination_data.iloc[index].to_numpy()
        }
        
    def __len__(self):
        return len(self.combination_data)
    
if __name__ == "__main__":
    dataset = TftDataset()
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    from models import TftEncoder, TftDecoder
    encoder = TftEncoder()
    decoder = TftDecoder()
    
    data = next(iter(train_dataloader))
    encoding = encoder(data)
    decoding = decoder(encoding)
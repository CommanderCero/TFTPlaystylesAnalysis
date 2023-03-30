import torch
import torch.nn as nn
import torch.nn.functional as F

from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

from tft_dataset import NUM_CHAMPIONS, NUM_ITEMS, NUM_STARS, CHAMPIONS_PER_ROW, ITEMS_PER_CHAMPION, COMBINATIONS_PER_ROW

from typing import Dict

class TftEncoder(BaseEncoder):
    def __init__(self,
        embedding_size:int = 128,
        champion_encoding_size:int = 64,
        item_encoding_size:int = 64,
        star_encoding_size:int = 16
    ):
        super().__init__()
        self.champion_embedding = nn.Embedding(NUM_CHAMPIONS, champion_encoding_size)
        self.item_embedding = nn.Embedding(NUM_ITEMS, item_encoding_size)
        self.star_embedding = nn.Embedding(NUM_STARS, star_encoding_size)
        
        num_inputs = CHAMPIONS_PER_ROW * champion_encoding_size
        num_inputs += CHAMPIONS_PER_ROW * ITEMS_PER_CHAMPION * item_encoding_size
        num_inputs += CHAMPIONS_PER_ROW * star_encoding_size
        num_inputs += COMBINATIONS_PER_ROW
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        self.embedding = nn.Linear(512, embedding_size)
        self.log_var = nn.Linear(512, embedding_size)
    
    def forward(self, x: Dict[str, torch.Tensor]):
        champion_embeddings = self.champion_embedding(x["champions"])
        item_embedding = self.item_embedding(x["champion_items"])
        star_embeddings = self.star_embedding(x["champion_star"])
        
        encoder_inp = torch.concat([
            champion_embeddings.reshape(champion_embeddings.shape[0], -1),
            item_embedding.reshape(item_embedding.shape[0], -1),
            star_embeddings.reshape(star_embeddings.shape[0], -1),
            x["combinations"]
        ], dim=1)
        
        encoding = self.encoder(encoder_inp)
        return self.embedding(encoding)
    
class TftDecoder(BaseDecoder):
    def __init__(self,
        embedding_size:int = 128
    ):
        super().__init__()
        num_outputs = CHAMPIONS_PER_ROW * NUM_CHAMPIONS
        num_outputs += CHAMPIONS_PER_ROW * ITEMS_PER_CHAMPION * NUM_ITEMS
        num_outputs += CHAMPIONS_PER_ROW * NUM_STARS
        num_outputs += COMBINATIONS_PER_ROW
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.decoder(x)
        start = 0
        
        champions = x[:, start:start+CHAMPIONS_PER_ROW*NUM_CHAMPIONS]
        start += CHAMPIONS_PER_ROW*NUM_CHAMPIONS
        
        items = x[:, start:start+ITEMS_PER_CHAMPION*CHAMPIONS_PER_ROW*NUM_ITEMS]
        start += ITEMS_PER_CHAMPION*CHAMPIONS_PER_ROW*NUM_ITEMS
        
        champion_stars = x[:, start:start+CHAMPIONS_PER_ROW*NUM_STARS]
        start += CHAMPIONS_PER_ROW*NUM_STARS
        
        combinations = x[:, start:]
        
        return {
            "champions": champions.reshape(-1, CHAMPIONS_PER_ROW, NUM_CHAMPIONS),
            "champion_items": items.reshape(-1, ITEMS_PER_CHAMPION*CHAMPIONS_PER_ROW, NUM_ITEMS),
            "champion_star": champion_stars.reshape(-1, CHAMPIONS_PER_ROW, NUM_STARS),
            "combinations": combinations
        }


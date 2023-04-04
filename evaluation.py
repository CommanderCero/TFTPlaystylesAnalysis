import numpy as np
import pandas as pd
import torch
import json

from models import TftEncoder, TftDecoder
from tft_dataset import TftDataset
from torch.utils.data import DataLoader


# Load dataset and model
dataset = TftDataset()
data_loader = DataLoader(dataset, batch_size=256)

encoder = TftEncoder(embedding_size=32)
encoder.load_state_dict(torch.load("results/encoder.pt", map_location=torch.device("cpu")))

# Compute embedding for all of our data
outputs = np.empty((len(dataset), 32))
for i, data in enumerate(iter(data_loader)):
    encoder_output = encoder(data)
    outputs[i * 256: i * 256 + len(encoder_output)] = encoder_output.detach().numpy()

embedding_df = pd.DataFrame(outputs)
embedding_df.to_csv("results/encoder_result.csv")

# Extract embedding for champions and items
with open("data/champion_index.json") as f:
    champions = json.load(f)
with open("data/items_index.json") as f:
    items = json.load(f)

def create_embedding_df(id_json, embedding: torch.nn.Embedding):
    ids = [value for name, value in id_json.items()]
    embeddings = embedding(torch.LongTensor(ids))
    df = pd.DataFrame(embeddings.detach().numpy(), index=ids)
    return df

champions_df = create_embedding_df(champions, encoder.champion_embedding)
champions_df.to_csv("results/champions_embedding.csv")

items_df = create_embedding_df(items, encoder.item_embedding)
items_df.to_csv("results/items_embedding.csv")


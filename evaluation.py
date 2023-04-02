import numpy as np
import pandas as pd
import torch
from models import TftEncoder, TftDecoder
from tft_dataset import TftDataset
from torch.utils.data import DataLoader


dataset = TftDataset()
data_loader = DataLoader(dataset, batch_size=256)

encoder = TftEncoder(embedding_size=32)

encoder.load_state_dict(torch.load("trained_model/encoder.pt"))

outputs = np.empty((len(dataset), 32))

for i, data in enumerate(iter(data_loader)):
    encoder_output = encoder(data)
    outputs[i * 256: i * 256 + len(encoder_output)] = encoder_output.detach().numpy()


df = pd.DataFrame(outputs)
df.to_csv("encoder_result.csv")


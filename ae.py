import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import sys

from tft_dataset import TftDataset
from models import TftEncoder, TftDecoder

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.encoder = TftEncoder(embedding_size=args.embedding_size)
        self.decoder = TftDecoder(args.embedding_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class AE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()
        self.train_loader = torch.utils.data.DataLoader(self.data, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.data, batch_size=64, shuffle=True)

        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _init_dataset(self):
        self.data = TftDataset()

    def mse_loss(self, recon_x, x):
        return F.mse_loss(
            recon_x.reshape(x.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="mean",
        )
        
    def ce_loss(self, recon_x, x):
        return F.cross_entropy(
            recon_x.reshape(-1, recon_x.shape[-1]),
            x.reshape(-1),
            reduction="mean"
        )

    def loss_function(self, recon_x, x):
        recon_loss = self.ce_loss(recon_x["champions"], x["champions"])
        recon_loss += self.ce_loss(recon_x["champion_items"], x["champion_items"])
        recon_loss += self.ce_loss(recon_x["champion_star"], x["champion_star"])
        recon_loss += self.mse_loss(recon_x["combinations"], x["combinations"])
        recon_loss /= 4
        return recon_loss

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            for key, value in data.items():
                data[key] = value.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        
        
if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(
    description='Main function to call training for different AutoEncoders')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                        help='Where to store images')
    parser.add_argument('--model', type=str, default='AE', metavar='N',
                        help='Which architecture to use')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='Which dataset to use')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    ae = AE(args)
    if __name__ == "__main__":
        try:
            os.stat(args.results_path)
        except :
            os.mkdir(args.results_path)
    
        try:
            autoenc = ae
        except KeyError:
            print('---------------------------------------------------------')
            print('Model architecture not supported. ', end='')
            print('Maybe you can implement it?')
            print('---------------------------------------------------------')
            sys.exit()
    
        try:
            for epoch in range(1, args.epochs + 1):
                autoenc.train(epoch)
                #autoenc.test(epoch)
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")
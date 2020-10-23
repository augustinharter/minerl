import torch as T
import torch.nn as nn
import torch.nn.functional as F
import utils
from itertools import chain

class AutoEncoder(nn.Module):
    def __init__(self, width, enc_dim, colorchs, activation=nn.Tanh):
        super().__init__()
        self.width = width
        self.enc = nn.Sequential(
            nn.Linear(colorchs*width*width, 32),
            activation(),
            nn.Linear(32, 16),
            activation(),
            nn.Linear(16, enc_dim),
            activation()
        )
        self.dec = nn.Sequential(
            nn.Linear(enc_dim, 16),
            activation(),
            nn.Linear(16, 32),
            activation(),
            nn.Linear(32, colorchs*width*width)
        )
        self.opti = T.optim.Adam(chain(self.enc.parameters(), self.dec.parameters()))

    def forward(self, x:T.Tensor):
        shape = x.shape
        x = x.flatten(start_dim=1)
        enc = self.enc(x)
        x = self.dec(enc)
        x = x.view(shape)
        return x, enc

    def convert_enc(self, enc):
        return (enc+1)/2

    def train_batch(self, batch):
        res, enc = self.forward(batch) # Forward pass

        loss = F.binary_cross_entropy_with_logits(res, batch) # Loss
        self.opti.zero_grad()
        loss.backward()
        self.opti.step()

        return loss.item(), T.sigmoid(res.detach()), enc.detach()

    def test_batch(self, batch):
        with T.no_grad():
            res, enc = self.forward(batch) # Forward pass()

        return T.sigmoid(res.detach()), enc.detach()


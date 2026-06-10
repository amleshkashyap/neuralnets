import torch.nn as nn

from transformer.gfg.PositionWiseFeedForward import PositionWiseFeedForward
from transformer.gfg.MultiHeadAttention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, dModel, numHeads, dFf, dropout):
        super(Encoder, self).__init__()
        self.selfAttention = MultiHeadAttention(dModel, numHeads)
        self.feedForward = PositionWiseFeedForward(dModel, dFf)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attentionOutput = self.selfAttention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attentionOutput))
        feedForwardOutput = self.feedForward(x)
        x = self.norm2(x + self.dropout(feedForwardOutput))
        return x
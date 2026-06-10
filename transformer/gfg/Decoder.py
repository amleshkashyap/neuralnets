import torch.nn as nn

from transformer.gfg.MultiHeadAttention import MultiHeadAttention
from transformer.gfg.PositionWiseFeedForward import PositionWiseFeedForward


class Decoder(nn.Module):
    def __init__(self, dModel, numHeads, dFf, dropout):
        super(Decoder, self).__init__()
        self.selfAttention = MultiHeadAttention(dModel, numHeads)
        self.crossAttention = MultiHeadAttention(dModel, numHeads)
        self.feedForward = PositionWiseFeedForward(dModel, dFf)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.norm3 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoderOutput, sourceMask, targetMask):
        attentionOutput = self.selfAttention(x, x, x, targetMask)
        x = self.norm1(x + self.dropout(attentionOutput))
        attentionOutput = self.crossAttention(x, encoderOutput, encoderOutput, sourceMask)
        x = self.norm2(x + self.dropout(attentionOutput))
        feedForwardOutput = self.feedForward(x)
        x = self.norm3(x + self.dropout(feedForwardOutput))
        return x
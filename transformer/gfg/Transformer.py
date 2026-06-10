import torch
import torch.nn as nn

from transformer.gfg.PositionalEncoding import PositionalEncoding
from transformer.gfg.Encoder import Encoder
from transformer.gfg.Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, sourceVocabSize, targetVocabSize, dModel, numHeads, numLayers, dFf, maxSeqLength, dropout):
        super(Transformer, self).__init__()
        self.encoderEmbedding = nn.Embedding(sourceVocabSize, dModel)
        self.decoderEmbedding = nn.Embedding(targetVocabSize, dModel)
        self.positionalEncoding = PositionalEncoding(dModel, maxSeqLength)

        self.encoderLayers = nn.ModuleList([Encoder(dModel, numHeads, dFf, dropout) for _ in range(numLayers)])
        self.decoderLayers = nn.ModuleList([Decoder(dModel, numHeads, dFf, dropout) for _ in range(numLayers)])

        self.fc = nn.Linear(dModel, targetVocabSize)
        self.dropout = nn.Dropout(dropout)

    def generateMask(self, source, target):
        sourceMask = (source != 0).unsqueeze(1).unsqueeze(2)
        targetMask = (target != 0).unsqueeze(1).unsqueeze(3)
        seqLength = target.size(1)
        noPeakMask = (1 - torch.triu(torch.ones(1, seqLength, seqLength), diagonal = 1)).bool()
        targetMask = targetMask & noPeakMask
        return sourceMask, targetMask

    def forward(self, source, target):
        sourceMask, targetMask = self.generateMask(source, target)
        sourceEmbedded = self.dropout(self.positionalEncoding(self.encoderEmbedding(source)))
        targetEmbedded = self.dropout(self.positionalEncoding(self.decoderEmbedding(target)))

        encoderOutput = sourceEmbedded
        for encoderLayer in self.encoderLayers:
            encoderOutput = encoderLayer(encoderOutput, sourceMask)

        decoderOutput = targetEmbedded
        for decoderLayer in self.decoderLayers:
            decoderOutput = decoderLayer(decoderOutput, encoderOutput, sourceMask, targetMask)

        return self.fc(decoderOutput)
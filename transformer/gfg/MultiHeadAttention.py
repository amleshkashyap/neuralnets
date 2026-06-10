import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dModel, numHeads):
        super(MultiHeadAttention, self).__init__()
        assert dModel % numHeads == 0, "dModel must be divisible by numHeads"

        self.dModel = dModel
        self.numHeads = numHeads
        self.dK = dModel // numHeads
        # get the query, key, value tensors required for the transformer
        self.Wq = nn.Linear(dModel, dModel)
        self.Wk = nn.Linear(dModel, dModel)
        self.Wv = nn.Linear(dModel, dModel)
        self.Wo = nn.Linear(dModel, dModel)

    def scaledDotProductAttention(self, Q, K, V, mask = None):
        attentionScores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dK)
        if mask is not None:
            attentionScores = attentionScores.masked_fill(mask == 0, -1e9)

        attentionProbs = torch.softmax(attentionScores, dim=-1)
        output = torch.matmul(attentionProbs, V)
        return output

    def splitHeads(self, x):
        batchSize, seqLength, dModel = x.size()
        return x.view(batchSize, seqLength, self.numHeads, self.dK).transpose(1, 2)

    def combineHeads(self, x):
        batchSize, _, seqLength, dK = x.size()
        return x.transpose(1, 2).contiguous().view(batchSize, seqLength, self.dModel)

    def forward(self, Q, K, V, mask = None):
        Q = self.splitHeads(self.Wq(Q))
        K = self.splitHeads(self.Wk(K))
        V = self.splitHeads(self.Wv(V))

        attentionOutput = self.scaledDotProductAttention(Q, K, V, mask = mask)
        return self.Wo(self.combineHeads(attentionOutput))
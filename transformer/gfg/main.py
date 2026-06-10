import torch
import torch.nn as nn
import torch.optim as optim

from transformer.gfg.Transformer import Transformer

if __name__ == "__main__":
    sourceVocabSize = 5000
    targetVocabSize = 5000
    dModel = 512
    numHeads = 8
    numLayers = 6
    dFf = 2048
    maxSeqLength = 100
    dropout = 0.1

    transformer = Transformer(sourceVocabSize, targetVocabSize, dModel, numHeads, numLayers, dFf, maxSeqLength, dropout)

    sourceData = torch.randint(1, sourceVocabSize, (64, maxSeqLength))
    targetData = torch.randint(1, targetVocabSize, (64, maxSeqLength))

    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = optim.Adam(transformer.parameters(), lr = 0.0001, betas = (0.9, 0.98), eps = 1e-9)

    transformer.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = transformer(sourceData, targetData[:, :-1])
        loss = criterion(output.contiguous().view(-1, targetVocabSize), targetData[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    transformer.eval()
    valSourceData = torch.randint(1, sourceVocabSize, (64, maxSeqLength))
    valTargetData = torch.randint(1, targetVocabSize, (64, maxSeqLength))

    with torch.no_grad():
        valOutput = transformer(valSourceData, valTargetData[:, :-1])
        valLoss = criterion(valOutput.contiguous().view(-1, targetVocabSize), valTargetData[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {valLoss.item()}")
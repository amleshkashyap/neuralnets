import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
import numpy as np
import random

random.seed(1)
torch.manual_seed(1)

class EncoderDecoder(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize = 1):
        super(EncoderDecoder, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.encoder = Encoder(self.inputSize, self.hiddenSize)
        self.decoder = Decoder(
            self.inputSize,
            self.hiddenSize,
            self.outputSize
        )

    def trainModel(self,
              XTrain,
              YTrain,
              epochs,
              targetLen,
              criterion,
              optimizer,
              method = 'recursive',
              tfr = 0.5,
              lr = 0.01,
              dynamicTf = False
    ):
        losses = np.full(epochs, np.nan)
        for epoch in range(epochs):
            predicted = torch.zeros(
                targetLen,
                XTrain.shape[1],
                XTrain.shape[2]
            )
            optimizer.zero_grad()
            _, encoderH = self.encoder(XTrain)
            decoderInput = XTrain[-1, :, :]
            decoderH = encoderH

            # recursive training - output of previous iteration is fed to next
            if method == 'recursive':
                for t in range(targetLen):
                    decoderOutput, decoderH = self.decoder(decoderInput, decoderH)
                    predicted[t] = decoderOutput
                    decoderInput = decoderOutput
            # teacher forcing - output of previous iteration will be fed to next with
            #   probability tfr, and actual output will be fed with (1 - tfr) - across epochs
            elif method == 'teacherForcing':
                if random.random() < tfr:
                    for t in range(targetLen):
                        decoderOutput, decoderH = self.decoder(decoderInput, decoderH)
                        predicted[t] = decoderOutput
                        decoderInput = YTrain[t, :, :]
                else:
                    for t in range(targetLen):
                        decoderOutput, decoderH = self.decoder(decoderInput, decoderH)
                        predicted[t] = decoderOutput
                        decoderInput = decoderOutput
            # mixed teacher forcing - teacher forcing but in the same epoch
            elif method == 'mixedTeacherForcing':
                for t in range(targetLen):
                    decoderOutput, decoderH = self.decoder(decoderInput, decoderH)
                    predicted[t] = decoderOutput
                    if random.random() < tfr:
                        decoderInput = YTrain[t, :, :]
                    else:
                        decoderInput = decoderOutput

            loss = criterion(predicted, YTrain)
            loss.backward()
            optimizer.step()
            losses[epoch] = loss.item()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}: {round(loss.item(), 4)}')

            # dynamic teacher forcing - empirically useful
            if dynamicTf and tfr > 0:
                tfr -= 0.02

        return losses

    def predict(self, x, targetLen):
        y = torch.zeros(
            targetLen,
            x.shape[1],
            x.shape[2]
        )
        _, encoderH = self.encoder(x)
        decoderInput = x[-1, :, :]
        decoderH = encoderH
        for t in range(targetLen):
            decoderOutput, decoderH = self.decoder(decoderInput, decoderH)
            y[t] = decoderOutput
            decoderInput = decoderOutput
        return y
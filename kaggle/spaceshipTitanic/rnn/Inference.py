import csv
import pandas as pd
import os
import torch
import numpy as np

class Inference:
    def __init__(self, model, criterion, bestResPath):
        self.model = model
        self.criterion = criterion
        self.bestResPath = os.getcwd()
        self.bestResPath = os.path.join(self.bestResPath, bestResPath)
        self.bestLabels = pd.read_csv(self.bestResPath)
        self.bestLabels.drop(
            'PassengerId',
            axis = 1,
            inplace = True
        )
        self.bestTensor = torch.tensor(
            self.bestLabels['Transported'].values,
            dtype = torch.float32
        ).reshape(-1, 1)

    def predict(self, xTensor, yTensor, mode = 'test'):
        # Deactivate dropout or batch normalization layers
        self.model.eval()
        yTensor = yTensor.reshape(-1, 1)

        # don't track gradients
        with torch.no_grad():
            # Forward pass through the model to make predictions (logits)
            rawLogits, _ = self.model(xTensor)

            # validation loss using the selected loss function
            valLoss = self.criterion(
                rawLogits,
                yTensor
            ).item()

            # get probabilities
            probabilities = torch.sigmoid(rawLogits)

            print(
                "Raw Probabilities (First 10):", probabilities[:10].squeeze().tolist()
            )
            print("True Labels (First 10):       ", yTensor[:10].squeeze().tolist())

            # use threshold of 0.5 to assign binary labels
            predictions = (probabilities >= 0.5).float()

        # convert back to numpy arrays
        yTrue = yTensor.numpy()
        yPred = predictions.numpy()

        # Calculate Accuracy
        correctPredictions = (yPred == yTrue).sum()
        totalSamples = len(yTrue)
        accuracy = (correctPredictions / totalSamples) * 100
        if mode == 'test':
            yBest = self.bestTensor.numpy()
            correctPredictionsBest = (yBest == yPred).sum()
            accuracyBest = (correctPredictionsBest / totalSamples) * 100
        else:
            accuracyBest = 0

        print("--- Validation Results ---")
        print(f"Validation Loss: {valLoss:.4f}")
        print(f"Accuracy:        {accuracy:.2f}%")
        print(f"Accuracy WRT Best:    {accuracyBest:.2f}%")

        np.savetxt(
            "results.csv",
            yPred.astype(bool),
            delimiter = ',',
            fmt = '%s'
        )
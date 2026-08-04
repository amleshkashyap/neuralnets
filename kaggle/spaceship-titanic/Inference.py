import csv

import torch
import numpy as np

class Inference:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def predict(self, df, testLabels):
        # Deactivate dropout or batch normalization layers
        self.model.eval()

        # convert the test dataset to numpy array
        xFeatures = df.to_numpy(dtype = np.float32)  # Shape: (testSampleSize, 15)

        xTensor = torch.tensor(
            xFeatures,
            dtype = torch.float32
        ).unsqueeze(1)

        yTensor = torch.tensor(
            testLabels,
            dtype = torch.float32
        ).reshape(-1, 1)

        # don't track gradients
        with torch.no_grad():
            # Forward pass through the model to make predictions (logits)
            rawLogits = self.model(xTensor)

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

        print("--- Validation Results ---")
        print(f"Validation Loss: {valLoss:.4f}")
        print(f"Accuracy:        {accuracy:.2f}%")

        np.savetxt(
            "results.csv",
            yPred.astype(bool),
            delimiter = ',',
            fmt = '%s'
        )
import torch
import numpy as np

class Inference:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def predict(self, df, testLabels):
        # Deactivate dropout or batch normalization layers
        self.model.eval()

        # Assume 'test_df' is your test DataFrame with 15 feature columns
        # Assume 'test_labels' is your NumPy array or Series of true labels
        xFeatures = df.values.astype(np.float32)  # Shape: (num_test_samples, 15)

        # Convert to 3D Tensor: (num_test_samples, 1, 15)
        xTensor = torch.tensor(
            xFeatures,
            dtype = torch.float32
        ).unsqueeze(1)

        yTensor = torch.tensor(
            testLabels,
            dtype = torch.float32
        ).reshape(-1, 1)

        # 2. Generate Predictions without Tracking Gradients
        with torch.no_grad():
            # Forward pass through the model
            rawLogits = self.model(xTensor)

            # Calculate validation loss using the training criterion
            valLoss = self.criterion(
                rawLogits,
                yTensor
            ).item()

            # Convert raw outputs (logits) into probabilities using Sigmoid
            probabilities = torch.sigmoid(rawLogits)

            # Convert probabilities to binary classes (0 or 1) using 0.5 threshold
            predictions = (probabilities >= 0.5).float()

        # 3. Calculate and Print Performance Metrics
        # Convert Tensors back to numpy arrays for calculation if needed
        yTrue = yTensor.numpy()
        yPred = predictions.numpy()

        # Calculate Accuracy
        correctPredictions = (yPred == yTrue).sum()
        totalSamples = len(yTrue)
        accuracy = (correctPredictions / totalSamples) * 100

        print("--- Validation Results ---")
        print(f"Validation Loss: {valLoss:.4f}")
        print(f"Accuracy:        {accuracy:.2f}%")

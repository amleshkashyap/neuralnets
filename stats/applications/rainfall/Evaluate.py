import random
from sklearn.metrics import balanced_accuracy_score

class Evaluate:
    def __init__(self, model):
        self.model = model

    def evaluate(self, XTest, YTest):
        self.model.eval()
        probabilities = self.model(XTest)
        prediction = probabilities.data.max(1, keepdim = True)[1].view(-1).tolist()

        # other predictions
        noRainPred = [0] * len(YTest)
        noSunPred = [1] * len(YTest)
        coinFlipPred = [random.randint(0, 1) for _ in range(len(YTest))]
        yestPred = XTest[:, -1, -1].view(-1).tolist()

        # scores
        baScore = balanced_accuracy_score
        modelScore = round(baScore(YTest, prediction), 4)
        noRainScore = round(baScore(YTest, noRainPred), 4)
        noSunScore = round(baScore(YTest, noSunPred), 4)
        coinFlipScore = round(baScore(YTest, coinFlipPred), 4)
        yestScore = round(baScore(YTest, yestPred), 4)

        # display
        print(f'Model Prediction Score: {modelScore}')
        print(f'No Rain Prediction Score: {noRainScore}')
        print(f'No Sun Prediction Score: {noSunScore}')
        print(f'Coin Flip Prediction Score: {coinFlipScore}')
        print(f'Yesterday Prediction Score: {yestScore}')
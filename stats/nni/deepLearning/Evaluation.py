import nni
import torch

class Evaluation:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def evaluate(self, XValidate, XTest, YTest):
        self.model.eval()
        _, hList = self.model(XValidate)
        h = (hList[-1, :]).unsqueeze(-2)
        predicted = []
        for seq in XTest.tolist():
            x = torch.tensor(data = [seq])
            y, h = self.model(x, h.unsqueeze(-2))
            predicted.append(y)
        testLoss = self.criterion(
            torch.tensor(data = predicted),
            YTest.view(-1)).item(
        )
        nni.report_final_result(testLoss)
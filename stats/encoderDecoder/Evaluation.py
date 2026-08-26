import matplotlib.pyplot as plt
import random

random.seed(1)

class Evaluation:
    def __init__(self, model):
        self.model = model

    def evaluate(self, XTest, YTest, tsTargetLength, testDsLength):
        self.model.eval()
        predicted = self.model.predict(XTest, tsTargetLength)

        fig, ax = plt.subplots(nrows = 3, ncols = 1)
        fig.set_size_inches(7.5, 6)

        for col in ax:
            r = random.randint(0, testDsLength)
            inputSeq = XTest[:, r, :].view(-1).tolist()
            targetSeq = YTest[:, r, :].view(-1).tolist()
            predSeq = predicted[:, r, :].view(-1).tolist()
            xAxis = range(len(inputSeq) + len(targetSeq))
            col.set_title(f'Test Sample: {r}')
            col.axis('off')
            col.plot(
                xAxis[:],
                inputSeq + targetSeq,
                color = 'blue'
            )
            col.plot(
                xAxis[len(inputSeq):],
                predSeq,
                label = 'Predicted',
                color = 'orange',
                linewidth = 3
            )
            col.vlines(
                len(inputSeq),
                0,
                6,
                color = 'grey'
            )
            col.legend(loc = 'upper right')
        plt.savefig('prediction.png')
        plt.show()
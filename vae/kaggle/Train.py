import pandas as pd
import torch
import time
import numpy as np

from vae.kaggle.FileUtils import FileUtils
from tqdm import tqdm

class Train:
    @staticmethod
    def trainLoop(
            model,
            lossFunction,
            trainLoader,
            testLoader = None,
            valLoader = None,
            scoreFuncs = None,
            epochs = 50,
            device = "cpu",
            optimizer = None,
            lrSchedule = None,
            checkpointFile = None,
            keep = True
    ):
        '''
        Train neural network from scratch - calculates a score every epoch with the given score functions
        '''
        if scoreFuncs is None:
            scoreFuncs = {}

        toTrack = ["epoch", "total time", "train loss", "lr"]
        if valLoader is not None:
            toTrack.append("val loss")

        if testLoader is not None:
            toTrack.append("test loss")

        for evalScore in scoreFuncs:
            toTrack.append("train" + evalScore)
            if valLoader is not None:
                toTrack.append("val" + evalScore)
            if testLoader is not None:
                toTrack.append("test" + evalScore)

        totalTrainTime = 0
        results = {}

        for item in toTrack:
            results[item] = []

        deleteOptimizer = False
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
            deleteOptimizer = True

        model.to(device)

        for epoch in tqdm(range(epochs), desc = "Epoch", leave = keep):
            model = model.train()
            Train.runEpoch(
                model,
                optimizer,
                trainLoader,
                lossFunction,
                device,
                results,
                scoreFuncs,
                prefix = 'train',
                desc = 'Training'
            )

            results["total time"].append(totalTrainTime)
            results["epoch"].append(epoch)
            results["lr"].append(optimizer.param_groups[0]["lr"])

            if valLoader is not None:
                model = model.eval()

                with torch.no_grad():
                    Train.runEpoch(
                        model,
                        optimizer,
                        valLoader,
                        lossFunction,
                        device,
                        results,
                        scoreFuncs,
                        prefix = 'val',
                        desc = 'Validation',
                        epoch = epoch
                    )

            if testLoader is not None:
                model = model.eval()
                with torch.no_grad():
                    Train.runEpoch(
                        model,
                        optimizer,
                        valLoader,
                        lossFunction,
                        device,
                        results,
                        scoreFuncs,
                        prefix = 'test',
                        desc = 'Testing',
                        epoch = epoch
                    )

            if lrSchedule is not None:
                if isinstance(lrSchedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lrSchedule.step(results["val loss"][-1])
                else:
                    lrSchedule.step()

            if checkpointFile is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'results': results
                }, checkpointFile)

        if deleteOptimizer:
            del optimizer

        return pd.DataFrame.from_dict(results)

    @staticmethod
    def runEpoch(
            model,
            optimizer,
            dataLoader,
            lossFunction,
            device,
            results,
            scoreFuncs,
            prefix = "",
            desc = None,
            keep = False,
            epoch = None
    ):
        '''
        Single epoch on training data and calculate loss
        Use that to perform backpropagation and update weights
        '''
        runningLoss = []
        yTrue = []
        yPred = []
        start = time.time()

        for inputs, labels in tqdm(dataLoader, desc = desc, leave = keep):
            inputs = FileUtils.moveTo(inputs, device)
            labels = FileUtils.moveTo(labels, device)

            yHat, mu, sigma = model(inputs)

            reconstructedLoss = lossFunction(yHat, labels)
            kdlElement = mu.pow(2).add_(sigma.exp()).mul_(-1).add_(1).add_(sigma)
            klDivergence = torch.sum(kdlElement).mul_(-0.5)
            loss = reconstructedLoss + klDivergence

            if model.training:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            runningLoss.append(loss.item())

            if len(scoreFuncs) > 0 and isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
                yHat = yHat.detch().cpu().numpy()
                yTrue.extend(labels.tolist())
                yPred.extend(yHat.tolist())

        end = time.time()
        yPred = np.asarray(yPred)

        # ie, it's a classification problem, else, regression problem
        if len(yPred.shape) == 2 and yPred.shape[1] > 1:
            yPred = np.argmax(yPred, axis = 1)

        results[prefix + ' loss'].append(np.mean(runningLoss))
        resultsStr = [f"{prefix} loss: {np.mean(runningLoss)}"]

        for name, scoreFunc in scoreFuncs.items():
            try:
                score = scoreFunc(yTrue, yPred)
                results[prefix + " " + name].append(score)
                resultsStr.append(f"{prefix} {name}: {score}")

                # if prefix == 'val' or prefix == 'test':
                    # checkpointer(score, (epoch + 1), model, optimizer)
            except Exception as e:
                results[prefix + " " + name].append(float("NaN"))

            print(" ".join(resultsStr))

        return end - start
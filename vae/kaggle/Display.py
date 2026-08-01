import matplotlib.pyplot as plt
import seaborn as sns

from torchsummary import summary
from torchview import draw_graph

class Display:
    @staticmethod
    def displayModelProperties(model):
        print(summary(model.cuda(), input_size=(1, 28, 28), batch_size=-1))

        modelGraph = draw_graph(model.cuda(), input_size=(1, 1, 28, 28), expand_nested=True)
        modelGraph.visual_graph

    @staticmethod
    def displayLabels(images, labels):
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=[13, 14], dpi=100)
        axes = axes.ravel()

        for i, (img, label) in enumerate(zip(images, labels)):
            axes[i].imshow(img.numpy()[0], cmap=plt.cm.gray)

            if i >= 15:
                break

        plt.show()

    @staticmethod
    def displayLoss(results):
        plt.figure(figsize=[12, 6], dpi=200)
        sns.lineplot(x='epoch', y='train loss', data=results, label='train')
        sns.lineplot(x='epoch', y='val loss', data=results, label='validation')
        plt.ylabel('binary cross entropy loss')
        plt.show()

    @staticmethod
    def displayInferenceResults(inputDigit, outs, reconImg):
        fig, axes = plt.subplots(nrows=2, ncols=6, layout='constrained', figsize=(14, 4))
        gridspec = axes[0, 0].get_subplotspec().get_gridspec()

        for a in axes[:, 0]:
            a.remove()

        for i, a in enumerate(axes[:, 1:].flat):
            a.axis('off')
            a.imshow(outs[i].detach().numpy()[0][0])

        subfig = fig.add_subfigure(gridspec[:, 0])
        axsLeft = subfig.subplots(1, 2, sharey=True)

        axsLeft[0].imshow(inputDigit[0][0])
        axsLeft[0].set_title('original', fontsize=12)
        axsLeft[1].imshow(reconImg[0][0])
        axsLeft[1].set_title('reconstructed', fontsize=12)

        axsLeft[0].axis('off')
        axsLeft[1].axis('off')

        subfig.suptitle('encoder-decoder', fontsize='x-large')
        fig.suptitle('generated samples', fontsize='xx-large')
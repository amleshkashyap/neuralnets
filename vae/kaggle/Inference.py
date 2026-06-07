import torch

class Inference:
    def __init__(self, model, device = 'cpu'):
        self.model = model.to(device)
        self.device = device

    def getDigits(self, dataset):
        # NOTE:
        idx = torch.randint(len(dataset), (1, ))
        return dataset[idx][0].unsqueeze(0)

    def getEncodings(self, image):
        with torch.no_grad():
            mu, sigma = self.model.encoder(image)

        return mu, sigma

    def getReconstructed(self, image):
        with torch.no_grad():
            recon, _, _ = self.model(image)

        return recon

    def infereence(self, *, dataset, nExamples = 1):
        out = []
        image = self.getDigits(dataset)
        mu, sigma = self.getEncodings(image)
        reconImage = self.getReconstructed(image)

        for example in range(nExamples):
            epsilon = torch.randn_like(mu)
            # NOTE:
            z = mu + sigma * epsilon
            out.append(self.model.decoder(z))

        return image, out, reconImage
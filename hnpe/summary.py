import torch
import torch.fft
from torch import nn


# it calculates the biased or unbiased autocorrelation function
# it is simply an implementation of the ACF in terms of pyTorch transformations
# note that there's no parameter to learn here
class AutocorrSeq(nn.Module):
    def __init__(self, n_lags, unbiased=False):
        super().__init__()

        self.n_lags = n_lags
        self.unbiased = unbiased

    def forward(self, X):

        if X.ndim == 1:
            X = X.view(1, 1, -1)
        else:
            X = X.view(len(X), 1, -1)

        # zero mean the signal
        Xavg = X.mean(dim=2)
        Xfeatures = X.clone()
        for i in range(len(Xfeatures)):
            Xfeatures[i, 0, :] = Xfeatures[i, 0, :] - Xavg[i]

        # loop across the lags
        n_tsp = X.shape[2]
        Xautocorr = torch.zeros(
            (len(X), self.n_lags + 1), dtype=torch.get_default_dtype()
        )
        for k in range(self.n_lags + 1):

            # cross product
            Xcross = Xfeatures[:, :, : n_tsp - k].mul(Xfeatures[:, :, k:])

            # takes the expected value
            if self.unbiased:
                denominator = n_tsp - k
            else:
                denominator = n_tsp
            Xaveraged = Xcross.sum(dim=2) / denominator
            Xautocorr[:, k] = Xaveraged[:, 0]

        Xautocorr = Xautocorr[:, 1:].div(
            Xautocorr[:, 0].view(len(Xautocorr), -1)
        )

        return Xautocorr


class YuleNet(nn.Module):
    def __init__(self, n_time_samples, n_features):
        super().__init__()

        # first layer
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=8, kernel_size=64,
            stride=1, padding=32, bias=True
        )
        self.relu1 = nn.ReLU()

        pooling1 = 16
        self.pool1 = nn.AvgPool1d(kernel_size=pooling1)

        # second layer
        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=8, kernel_size=64,
            stride=1, padding=32, bias=True
        )
        self.relu2 = nn.ReLU()

        pooling2 = int((n_time_samples // pooling1) // 16)
        self.pool2 = nn.AvgPool1d(kernel_size=pooling2)

        # dropout layer
        self.dropout = nn.Dropout(p=0.50)

        # fully connected layer
        self.linear = nn.Linear(
            in_features=8 * n_time_samples // (pooling1 * pooling2),
            out_features=n_features
        )
        self.relu3 = nn.ReLU()

    def forward(self, X):

        if X.ndim == 1:
            X = X.view(1, 1, -1)
        else:
            X = X.view(len(X), 1, -1)

        # 1D convolution along time
        Xconv1 = self.conv1(X)
        # non-linear activation
        Xrelu1 = self.relu1(Xconv1)
        # pooling
        Xpool1 = self.pool1(Xrelu1)

        # 1D convolution along time
        Xconv2 = self.conv2(Xpool1)
        # non-linear activation
        Xrelu2 = self.relu2(Xconv2)
        # pooling
        Xpool2 = self.pool2(Xrelu2)

        # flattening
        Xflatten = Xpool2.view(len(X), 1, -1)
        # dropout
        Xdropout = self.dropout(Xflatten)

        # fully connected layer
        Y = self.relu3(self.linear(Xdropout))

        return Y.view(len(X), -1)


# implementing the PEN neural network
# ref. https://arxiv.org/abs/1901.10230
class PEN(nn.Module):
    def __init__(self, d, n_features=2):
        super().__init__()

        self._d = d

        self.phi = nn.Sequential(
            nn.Linear(in_features=d + 1, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
        )

        self.rho = nn.Sequential(
            nn.Linear(in_features=10 + d, out_features=50),
            nn.Linear(in_features=50, out_features=50),
            nn.Linear(in_features=50, out_features=20),
            nn.Linear(in_features=20, out_features=n_features),
        )

    def forward(self, X):

        M = X.shape[1]
        mask = torch.stack([
            torch.roll(torch.eye(M, self._d + 1), shifts=i, dims=0)
            for i in range(M - self._d)
        ])
        Xshift = torch.matmul(X, mask).permute(1, 0, 2)
        Yphi = torch.sum(self.phi(Xshift), axis=1)
        Y = self.rho(torch.cat([X[:, : self._d], Yphi], dim=1))

        return Y


# Welch's method for power spectral density estimation
#
# TODO: clean this up with a full-pytorch implementation
class PowerSpecDens(nn.Module):
    def __init__(self, nbins=128, logscale=False):
        super().__init__()
        self.nbins = nbins
        self.logscale = logscale

    def forward(self, X):

        if X.ndim == 1:
            X = X.view(1, -1)
        else:
            X = X.view(len(X), -1)

        ntrials = X.shape[0]
        ns = X.shape[1]
        nfreqs = 2*(self.nbins-1) + 1
        windows = [wi + torch.arange(nfreqs) for wi in range(0,
                                                             ns-nfreqs+1,
                                                             int(nfreqs/2))]
        S = []
        for i in range(ntrials):
            Xi = X[i, :].view(-1)
            Si = []
            for w in windows:
                Si.append(torch.abs(torch.fft.rfft(Xi[w]))**2)
            Si = torch.mean(torch.stack(Si), axis=0)
            if self.logscale:
                Si = torch.log10(Si)
            S.append(Si)
        return torch.stack(S)

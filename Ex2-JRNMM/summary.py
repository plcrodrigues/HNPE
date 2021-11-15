import torch
from torch import nn
from scipy.signal import welch
from hnpe.summary import YuleNet, AutocorrSeq, PowerSpecDens
from functools import partial

torch.autograd.set_detect_anomaly(True)


class FourierLayer(nn.Module):
    def __init__(self, nfreqs=129):
        super().__init__()
        self.nfreqs = nfreqs

    def forward(self, X):
        if X.ndim == 1:
            X = X.view(1, -1)
        else:
            X = X.view(len(X), -1)
        Xfourier = []
        Xnumpy = X.clone().detach().to('cpu').numpy()
        _, Xfourier = welch(Xnumpy, nperseg=2*(self.nfreqs-1))
        return X.new_tensor(Xfourier)


class fourier_embedding(nn.Module):
    def __init__(self, d_out=1, n_time_samples=1024, plcr=True,
                 logscale=False):
        super().__init__()
        self.d_out = d_out
        if plcr:  # use my implementation for power spectral density
            self.net = PowerSpecDens(nbins=d_out, logscale=logscale)
        else:  # use implementation from scipy.signal.welch (much slower)
            self.net = FourierLayer(nfreqs=d_out)

    def forward(self, x):
        y = self.net(x.view(1, -1))
        return y


class autocorr_embedding(nn.Module):
    def __init__(self, d_out=1, n_time_samples=1024):
        super().__init__()
        self.d_out = d_out
        self.net = AutocorrSeq(n_lags=d_out)

    def forward(self, x):
        y = self.net(x.view(1, -1))
        return y


class yulenet_embedding(nn.Module):
    def __init__(self, d_out=1, n_time_samples=1024):
        super().__init__()
        self.d_out = d_out
        self.n_time_samples = n_time_samples
        self.net = YuleNet(n_features=d_out, n_time_samples=n_time_samples)

    def forward(self, x):
        y = self.net(x.reshape(-1, self.n_time_samples))
        return y


class identity(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

    def forward(self, x):
        return x


class debug_embedding(nn.Module):
    def __init__(self, d_out=1, n_time_samples=1024):
        super().__init__()
        self.net = nn.Linear(in_features=n_time_samples, out_features=d_out)

    def forward(self, x):
        y = self.net(x.view(1, -1))
        # A = torch.ones(33, 1024)
        # y = A * x
        return y


emb_dict = {}
emb_dict['Fourier'] = partial(fourier_embedding, plcr=True)
emb_dict['Autocorr'] = autocorr_embedding
emb_dict['YuleNet'] = yulenet_embedding
emb_dict['Debug'] = debug_embedding


class summary_JRNMM(nn.Module):

    def __init__(self, n_extra=0,
                 d_embedding=33,
                 n_time_samples=1024,
                 type_embedding='Fourier'):

        super().__init__()
        self.n_extra = n_extra
        self.n_time_samples = n_time_samples
        self.d_embedding = d_embedding
        self.embedding = emb_dict[type_embedding](
            d_out=d_embedding, n_time_samples=n_time_samples
        )

    def forward(self, x, n_extra=None):
        self.embedding.eval()
        if n_extra is None:
            n_extra = self.n_extra
        if x.ndim == 2:
            x = x[None, :, :]
        y = []
        for xi in x:
            yi = [self.embedding(xi[:, 0])]
            for j in range(n_extra):
                yi.append(self.embedding(xi[:, j+1]))
            yi = torch.cat(yi).T
            y.append(yi)
        y = torch.stack(y)
        return y

        # THIS CHANGE BROUGHT CONFLICTS TO THE CODE
        # x = x.transpose(1, 2).reshape(-1, self.n_time_samples)
        # y_ = self.embedding(x)
        # y_ = y_.view(
        #     -1, 1 + n_extra, self.d_embedding
        # ).transpose(1, 2)
        # # assert torch.allclose(y_, y)
        # # print('ok')
        # return y_


if __name__ == '__main__':
    n_extra = 9
    d_embedding = 33
    x = torch.randn(3, 1024, 10)
    net = summary_JRNMM(n_extra=n_extra,
                        d_embedding=d_embedding,
                        type_embedding='Fourier')
    y = net(x)

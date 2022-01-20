import torch

# Imports for the SBI package
from pyknos.nflows.distributions import base
from sbi.utils.get_nn_models import build_nsf
from sbi.utils.sbiutils import standardizing_net

class IdentityToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, n_extra=0):
        return x


class ToyModelFlow_naive_nflows_amortizeNextra(base.Distribution):

    def __init__(self, batch_theta, batch_x, embedding_net,
                 z_score_theta=True, z_score_x=True):

        super().__init__()

        embedding_net = torch.nn.Sequential(
            embedding_net
        )
        self._embedding_net = embedding_net

        # instantiate the flow
        flow = build_nsf(batch_x=batch_theta,
                         batch_y=batch_x, ## changed: no mean dim=1 needed
                         z_score_x=z_score_theta,
                         z_score_y=z_score_x,
                         embedding_net=embedding_net,
                         num_transforms=10)  # same capacity as factorized

        self._flow = flow

    def _log_prob(self, inputs, context):
        logp = self._flow.log_prob(inputs, context) # no mean dim=1 needed
        return logp

    def _sample(self, num_samples, context):
        samples = self._flow.sample(num_samples, context)[0] # no mean dim=1 needed
        return samples

    def save_state(self, filename):
        state_dict = {}
        state_dict['flow'] = self._flow.state_dict()
        torch.save(state_dict, filename)

    def load_state(self, filename):
        state_dict = torch.load(filename, map_location='cpu')
        self._flow.load_state_dict(state_dict['flow'])


def build_flow(batch_theta, batch_x, embedding_net=torch.nn.Identity(),
               naive=True, z_score_x=False):
    if naive:
        flow = ToyModelFlow_naive_nflows_amortizeNextra(batch_theta,
                                         batch_x,
                                         embedding_net,
                                         z_score_x=z_score_x)
    else:
        flow = None

    return flow

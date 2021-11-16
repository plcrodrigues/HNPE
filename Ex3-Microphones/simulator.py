from sbi.utils import BoxUniform
import torch
import numpy as np


class prior_SourceLocalization(BoxUniform):

    def __init__(self, low, high):
        super().__init__(low, high)
        self.low = low
        self.high = high

    def condition(self, u):
        """
        This functions returns the prior distribution just for the alpha
        parameter. It is written like this for compatibility purposes with
        the Pyro framework
        """
        return BoxUniform(low=self.low[:1], high=self.high[:1])


def simulator_SourceLocalization(theta,
                                 n_extra=0,
                                 global_coord=0,
                                 p_local=None,
                                 dout=10,
                                 sigma=0.0,
                                 rotation=0.0,
                                 output_locals=False):
    """Define the simulator function

    Parameters
    ----------
    theta : torchtensor, shape (n_trials, 2)
        ndarray of trials.
    n_extra : int
        how many extra observations sharing the same global-coordinate should
        we simulate. The minimum is 0, for which the output is simply that with
        theta. n_extra > 0 will generate other outputs with different
        local-coordinates's but the same global-coordinate. the first
        coordinate of the sampled observation is the one corresponding to the
        input theta
    p_local : torch.distribution
        probability distribution from which to sample the local-coordinate of
        the extra observation

    Returns
    -------
    x : torchtensor shape (n_trials, 1, 1+n_extra)
        observations for the model with different input parameters

    """

    if theta.ndim == 1:
        return simulator_SourceLocalization(theta.view(1, -1),
                                            n_extra,
                                            global_coord,
                                            p_local,
                                            dout,
                                            sigma,
                                            rotation,
                                            output_locals)

    # locations of the microphones
    m1 = torch.tensor([-np.cos(rotation), -np.sin(rotation)])
    m2 = torch.tensor([np.cos(rotation), np.sin(rotation)])

    x = []
    local_variables = []
    for thetai in theta:

        thetai = thetai.detach().clone()

        ITD = (torch.norm(thetai - m1) - torch.norm(thetai - m2)).item()
        if sigma == 0.0:
            xi = [torch.tensor(dout*[ITD]).view(1, -1)]
        else:
            xi = [torch.normal(ITD*torch.ones(dout),
                               torch.tensor([sigma])).view(1, -1)]
        if n_extra > 0:

            local_list = p_local.condition(
                thetai[global_coord]).sample((n_extra,))
            local_variables_i = []

            for j in range(n_extra):

                if global_coord == 0:
                    uj = thetai[0]
                    vj = local_list[j]
                elif global_coord == 1:
                    uj = local_list[j]
                    vj = thetai[1]

                thetaj = torch.tensor([uj, vj])
                local_variables_i.append(thetaj)
                ITD = (torch.norm(thetaj - m1) -
                       torch.norm(thetaj - m2)).item()

                if sigma == 0.0:
                    xj = [torch.tensor(dout*[ITD]).view(1, -1)]
                else:
                    xj = [torch.normal(ITD*torch.ones(dout),
                                       torch.tensor([sigma])).view(1, -1)]

                xi = xi + xj

            local_variables.append(torch.stack(local_variables_i))

        x.append(torch.cat(xi).T)

    if output_locals:
        return torch.stack(x), torch.stack(local_variables)
    else:
        return torch.stack(x)


def get_ground_truth(meta_parameters, p_local=None):
    "Take the parameters dict as input and output the observed data."

    theta = meta_parameters["theta"].clone()
    observation = simulator_SourceLocalization(theta,
                                               meta_parameters["n_extra"],
                                               meta_parameters["global_coord"],
                                               p_local,
                                               meta_parameters["n_sf"],
                                               meta_parameters["sigma"],
                                               meta_parameters["rotation"])

    # get the ground_truth observation data
    ground_truth = {}
    ground_truth["theta"] = meta_parameters["theta"].clone().detach()
    ground_truth["observation"] = observation

    return ground_truth

import torch
from sbi.utils import BoxUniform
from torch.distributions import Distribution, Categorical, constraints
from torch.distributions.constraints import Constraint


class prior_ToyModel(BoxUniform):
    def __init__(self, low, high):
        super().__init__(low, high)
        self.low = low
        self.high = high

    def condition(self, beta):
        """
        This functions returns the prior distribution just for the alpha
        parameter. It is written like this for compatibility purposes with
        the Pyro framework
        """
        return BoxUniform(low=self.low[:1], high=self.high[:1])

def simulator_ToyModel_amortizeNextra(theta, n_trials=1, p_alpha=None, p_nextra=None, gamma=1.0,
                       sigma=0.0, ground_truth=False, aggregate_method=None):

    if theta.ndim == 1:
        return simulator_ToyModel_amortizeNextra(theta.view(1, -1), n_trials,
                                  p_alpha, p_nextra, gamma, sigma, ground_truth)

    x = []
    for thetai in theta:
        thetai = thetai.detach().clone()
        alphai = thetai[0]
        betai = thetai[1]
        n_extrai = p_nextra.sample()
        x0_i = torch.tensor([list(alphai * (betai**gamma) + sigma*torch.randn(n_trials))])
        xn_i = torch.tensor([0.]).reshape(-1,1)
        if n_extrai > 0:
            alphaj_list = p_alpha.condition(betai).sample((n_extrai,))
            betaj = betai
            xn_i = torch.tensor([list(alphaj * (betaj**gamma) + sigma*torch.randn(
                n_trials)) for alphaj in alphaj_list])
        x.append(torch.cat([x0_i, xn_i], dim=0))

    if (not ground_truth):
        x = preprocess_for_amortizeNextra(x, aggregate_method)
    
    return x

def get_ground_truth(meta_parameters, p_alpha=None, p_nextra=None):
    "Take the parameters dict as input and output the observed data."

    theta = meta_parameters["theta"].clone()

    observation = simulator_ToyModel_amortizeNextra(theta, 
                                     n_trials=meta_parameters["n_trials"],
                                     p_alpha=p_alpha,
                                     p_nextra=p_nextra,
                                     gamma=meta_parameters["gamma"],
                                     sigma=0.0,
                                     ground_truth = True)
    
    # get the ground_truth observation data
    ground_truth = {}
    ground_truth["theta"] = meta_parameters["theta"].clone().detach()
    ground_truth["observation"] = preprocess_for_amortizeNextra(observation, meta_parameters["aggregate_method"])
    ground_truth["extra_obs"] = observation[0][1:]
    return ground_truth
    
def aggregate_extra_obs(xn, method='mean'):
    if method == 'mean':
        return xn.mean()
    elif method is None:
        return xn
    else:
        raise ValueError(f'Aggregation method "{method}" not implemented.')
    
def preprocess_for_amortizeNextra(x, aggregate_method='mean'):
    x0 = torch.stack([x[i][0] for i in range(len(x))])
    xn = [x[i][1:] for i in range(len(x))]

    x_new=[]
    for i in range(len(x)):
        xn_i = torch.tensor([aggregate_extra_obs(xn[i], method=aggregate_method)])
        if xn_i == 0:
            nextrai = torch.tensor([0.])
        else:
            nextrai = torch.tensor([xn[i].shape[0]])
        xi = torch.cat([x0[i],nextrai,xn_i])
        x_new.append(xi)

    x_new = torch.stack(x_new)

    return x_new



if __name__ == "__main__":

    from functools import partial

    meta_parameters = {}
    meta_parameters["theta"] = torch.tensor([0.5, 0.5])
    meta_parameters["gt_nextra"] = 10
    meta_parameters["gamma"] = 1.0
    meta_parameters["noise"] = 0.0
    meta_parameters["nextra_range"] = 40
    meta_parameters["aggregate_method"] = 'mean'

    nextra_probs = torch.ones(meta_parameters["nextra_range"])/meta_parameters["nextra_range"]
    prior_nextra = Categorical(nextra_probs)

    gt_nextra_prob = torch.zeros_like(nextra_probs)
    gt_nextra_prob[meta_parameters['gt_nextra']] = 1
    p_gt_nextra = Categorical(gt_nextra_prob)

    prior = prior_ToyModel(low=torch.tensor([0.0, 0.0]),
                           high=torch.tensor([1.0, 1.0]))

    meta_parameters["n_trials"] = 1
    ground_truth = get_ground_truth(meta_parameters, prior, p_gt_nextra)
    x = ground_truth['observation']
    xn = ground_truth['extra_obs']
    print(x)
    print(xn.shape)

    theta = prior.sample((100_000,))

    # choose how to setup the simulator
    simulator = partial(simulator_ToyModel_amortizeNextra,
                        n_trials=meta_parameters["n_trials"],
                        p_alpha=prior,
                        p_nextra=prior_nextra,
                        gamma=meta_parameters["gamma"],
                        sigma=meta_parameters["noise"],
                        aggregate_method=meta_parameters["aggregate_method"])

    x = simulator(theta)
    print(x.shape)
    
import torch
import numpy as np
from sbi.utils import BoxUniform

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects


# setup all parameters
JRNMM_parameters = {}
JRNMM_parameters['A'] = 3.25
JRNMM_parameters['B'] = 22.0
JRNMM_parameters['a'] = 100.0
JRNMM_parameters['b'] = 50.0
JRNMM_parameters['C'] = 135.0
JRNMM_parameters['C1'] = 1.00*JRNMM_parameters['C']
JRNMM_parameters['C2'] = 0.80*JRNMM_parameters['C']
JRNMM_parameters['C3'] = 0.25*JRNMM_parameters['C']
JRNMM_parameters['C4'] = 0.25*JRNMM_parameters['C']
JRNMM_parameters['vmax'] = 5
JRNMM_parameters['v0'] = 6
JRNMM_parameters['r'] = 0.56
JRNMM_parameters['mu'] = 220.0
JRNMM_parameters['s4'] = 0.01
JRNMM_parameters['sigma'] = 2000.0
JRNMM_parameters['s6'] = 1.00
JRNMM_parameters['gain'] = 0.00  # in db


class prior_JRNMM(BoxUniform):
    def __init__(self, parameters):
        self.parameters = parameters
        low = []
        high = []
        for i in range(len(parameters)):
            low.append(parameters[i][1])
            high.append(parameters[i][2])
        super().__init__(low=torch.tensor(low, dtype=torch.float32),
                         high=torch.tensor(high, dtype=torch.float32))

    def condition(self, gain):
        """
        This functions returns the prior distribution for [C, mu, sigma]
        parameter. It is written like this for compatibility purposes with
        the Pyro framework
        """

        low = []
        high = []
        for i in range(len(self.parameters)):
            if self.parameters[i][0] == 'gain':
                pass
            else:
                low.append(self.parameters[i][1])
                high.append(self.parameters[i][2])
        low = torch.tensor(low, dtype=torch.float32)
        high = torch.tensor(high, dtype=torch.float32)
        return BoxUniform(low=low, high=high)


def simulator_JRNMM(theta, input_parameters, t_recording, n_extra=0,
                    p_gain=None, fix_gain=False):
    """Define the simulator function

    Parameters
    ----------
    theta : torchtensor, shape (n_trials, dim_theta)
        ndarray of trials.
    n_extra : int
        how many extra observations sharing the same beta should we simulate.
        the minimum is 0, for which the output is simply that with theta.
        n_instances > 0 will generate other outputs with other [C, mu, sigma]
        but the same gain. the first coordinate of the sampled observation is
        the one corresponding to the input theta
    p_gain : torch.distribution
        probability distribution from which to sample the different values of
        [C, mu, sigma] for a given gain (only used when n_extra > 0)
    fix_gain : bool
        whether the gain parameter should remain fixed, meaning we that we do
        inference only on three parameters

    Returns
    -------
    x : torchtensor shape (n_trials, n_time_samples, 1+n_extra)
        observations for the model with different input parameters

    """

    if theta.ndim == 1:
        return simulator_JRNMM(theta.view(1, -1),
                               input_parameters,
                               t_recording,
                               n_extra,
                               p_gain,
                               fix_gain)

    x = []
    xextra = []

    for thetai in theta:

        JRNMM_parameters_i = JRNMM_parameters.copy()
        thetai = thetai.detach().clone().to("cpu").numpy().astype(np.float64)
        for i, p in enumerate(input_parameters):
            JRNMM_parameters_i[p] = thetai[i]
            if p == 'gain':
                gaini = thetai[i]

        xi = simulate_jansen_rit_StrangSplitting(t_recording,
                                                 JRNMM_parameters_i,
                                                 fix_gain=fix_gain)
        xi = xi - np.mean(xi)
        x.append(xi)

        xextrai = []
        for _ in range(n_extra):

            JRNMM_parameters_j = JRNMM_parameters.copy()
            gainj = gaini
            thetaj = p_gain.sample((1,)).view(-1)
            thetaj = thetaj.detach().clone().cpu().numpy().astype(np.float64)
            for i, p in enumerate(input_parameters):
                if p == 'gain':
                    JRNMM_parameters_j[p] = gainj
                else:
                    JRNMM_parameters_j[p] = thetaj[i]

            xj = simulate_jansen_rit_StrangSplitting(
                t_recording, JRNMM_parameters_j
            )
            xj = xj - np.mean(xj)

            xextrai.append(xj)

        xextrai = theta.new(xextrai).T

        xextra.append(xextrai)

    x = theta.new(x).unsqueeze(-1)
    if n_extra == 0:
        return x
    else:
        xextra = torch.stack(xextra)
        return torch.cat([x, xextra], dim=-1)


def simulate_jansen_rit_StrangSplitting(trecording, parameters,
                                        x0=None, fix_gain=False):

    if x0 is None:
        x0 = np.random.randn(6)

    # import the sdbmpABC package which had previously been installed on R
    sdbmp = importr('sdbmsABC')
    rchol = robjects.r['chol']
    rt = robjects.r['t']

    burnin = 2.0
    T = trecording  # time interval for the datasets
    h = 1/2**10  # time step (corresponds to Delta)

    # theta_true: parameters used to simulate the reference data
    sigma = float(parameters['sigma'])
    mu = float(parameters['mu'])
    C = float(parameters['C'])

    # fixed model coefficients
    A = float(parameters['A'])
    B = float(parameters['B'])
    a = float(parameters['a'])
    b = float(parameters['b'])
    v0 = float(parameters['v0'])
    vmax = float(parameters['vmax'])
    r = float(parameters['r'])
    s4 = float(parameters['s4'])
    s6 = float(parameters['s6'])

    if fix_gain:
        gain_db = float(0.0)
    else:
        gain_db = float(parameters['gain'])
    gain_abs = 10**(gain_db/10)

    startv = robjects.FloatVector(list(x0))
    grid = robjects.FloatVector(list(np.arange(0, T+burnin, h)))
    M1 = sdbmp.exp_matJR(h, a, b)
    M2 = rt(rchol(sdbmp.cov_matJR(
        h, robjects.FloatVector([0, 0, 0, s4, sigma, s6]), a, b
    )))
    X = gain_abs * np.array(sdbmp.Splitting_JRNMM_output_Cpp(
        h, startv, grid, M1, M2, mu, C, A, B, a, b, v0, r, vmax
    ))
    X = X[int(burnin/h):]
    X = X[::8]

    return X


def get_ground_truth(meta_parameters, input_parameters, p_gain=None):
    "Take the parameters dict as input and output the observed data."

    # ground truth observation
    signal = simulator_JRNMM(
        theta=meta_parameters["theta"],
        input_parameters=input_parameters,
        t_recording=meta_parameters["t_recording"],
        n_extra=meta_parameters["n_extra"],
        p_gain=p_gain
    )

    # get the ground_truth observation data
    ground_truth = {}
    ground_truth["theta"] = meta_parameters["theta"].clone().detach()
    ground_truth["observation"] = signal

    return ground_truth


if __name__ == "__main__":

    meta_parameters = {}
    meta_parameters["theta"] = torch.tensor([0.5, 0.5])
    meta_parameters["n_extra"] = 2
    meta_parameters["t_recording"] = 8
    prior = prior_JRNMM(parameters=[('C', 10.0, 250.0),
                                    ('gain', -10.0, +10.0)])
    theta = prior.sample((3,))
    x = simulator_JRNMM(
        theta, input_parameters=['C', 'gain'],
        t_recording=meta_parameters["t_recording"],
        n_extra=meta_parameters["n_extra"], p_gain=prior
    )

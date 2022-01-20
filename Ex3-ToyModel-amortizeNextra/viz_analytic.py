import torch
import numpy as np 

def get_proba(log_p):
    "Compute probability from log_prob in a safe way."
    if isinstance(log_p, np.ndarray):
        log_p = torch.tensor(log_p)
    log_p = log_p.to(dtype=torch.float64)
    log_p -= torch.logsumexp(log_p, dim=-1)
    return torch.exp(log_p)

def analytic_posterior(a, b, x_obs, xn, sig=1e-2):
    """Return the analytic posterior.
    Parameters
    ----------
    a, b : ndarray, shape (n_samples,)
        Value of the parameter to compute the posterior on.
    x_obs : ndarray, shape (1 + n_extra,) or (1, 1 + n_extra)
        Observed data for the toy model with n_extra points.
    sig : float
        tolerance for the dirac function.

    Returns
    -------
    p : ndarray, shape (n_samples)
        Value of the posterior at each point (a, b), normalized on the samples.
    """
    
    n = int(x_obs[1])

    mu = max(torch.cat([torch.tensor([x_obs[0]]).reshape(-1,1), xn]))

    support = (
        (0 <= a) * (a <= 1)
        * (mu <= b) * (b <= 1)
    )
    log_p = -(x_obs[0] - a * b)**2 / (2 * sig ** 2)
    if n > 0:
        log_p += (
            np.log(n) - n * np.log(b)
            # Do not include the normalization constant bellow as it make it
            # unstable and it is not useful to compute the distance.
            # - np.log(1 / mu ** n - 1)
        )
    else:
        log_p += -np.log(-np.log(x_obs[0]))
    log_p[~support | (log_p < -400)] = -np.inf

    p = get_proba(log_p)
    assert not any(np.isinf(p))
    return p

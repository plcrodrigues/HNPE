"""
Some useful functions to evaluate the learned posterior (flow) by comparing it to 
the analytic posterior and/or the dirac function at the ground-truth parameters. 

These functions enable to compute distances between distributions and plot results.
"""

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches

def get_proba(log_p):
    """Compute probability from log_prob in a safe way.
    
    Parameters
    ----------
    log_p: ndarray, shape (n_samples*n_samples,)
        Values of log_p for each point (a,b) of the samples (n_samples, n_samples).
    """
    if isinstance(log_p, np.ndarray):
        log_p = torch.tensor(log_p)
    log_p = log_p.to(dtype=torch.float64)
    log_p -= torch.logsumexp(log_p, dim=-1)
    return torch.exp(log_p)

def analytic_posterior(a, b, x_obs, sig=1e-2):
    """ Return the analytic posterior.

    Parameters
    ----------
    a, b : ndarray, shape (n_samples,)
        Value of the parameter to compute the posterior on.
    x_obs : ndarray, shape (1 + n_extra,)
        Observed data for the toy model with n_extra points.
    sig : float
        Tolerance for the dirac function.

    Returns
    -------
    p : ndarray, shape (n_samples)
        Value of the posterior at each point (a, b), normalized on the samples.
    """
    n = x_obs.shape[0]-1

    mu = mu = max(x_obs)

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


def get_grid(n_samples=1000):
    """Return grid for evaluation.
    
    Parameters
    ----------
    n_samples : int 
        Number of samples, number of points in linspace.
    
    Returns 
    -------
    samples : torch.Tensor, shape (n_samples*n_samples, 2)
    XX, YY : torch.Tensor, shape (n_samples, n_samples)
    """
    t = torch.linspace(1 / n_samples, 1, n_samples)
    XX, YY = torch.meshgrid(t, t)
    samples = torch.cat([XX.reshape(-1, 1), YY.reshape(-1, 1)], dim=1)

    return samples, XX, YY

def get_posteriors_on_grid(posterior, ground_truth, samples, filename='posterior_on_grid.pkl'):
    """Compute posterior values on grid returned by the function get_grid() above.
    
    Parameters
    ----------
    posterior : DirectPosterior object from sbi.inference.posteriors
    ground_truth : dict with a key 'observation' pointing to a ndarray of shape (1 + n_extra,).
    samples : torch.Tensor, shape (n_samples*n_samples, 2)
        Output of the get_grid() function above.
    filename : str 
    
    Returns
    -------
    p_learned, p_ana : ndarray, shape (n_samples)
        Value of the posterior at each point (a, b) of the samples.
    """
    # Learned posterior
    log_p = posterior.log_prob(samples)
    torch.save(log_p, filename)
    p_learned = get_proba(log_p)
    assert p_learned.dtype == torch.float64

    # Analytic posterior
    x_obs = ground_truth['observation'][0]
    a, b = samples.numpy().T
    p_ana = analytic_posterior(a, b, x_obs[0].numpy(), sig=1e-2)
    assert p_ana.dtype == torch.float64

    return p_learned, p_ana


def plot_truevslearned_2d_pdf_contours(p_ana, p_learned, theta, XX, YY, title=None):
    """Plot analytic and learned posterior on grid returned by the function get_grid() above.
    
    Parameters
    ----------
    p_ana, p_learned : ndarray, shape (n_samples)
        Output of the get_posteriors_on_grid() function above.
    theta : ndarray, shape (2,)
        Values of the ground-truth parameters [alpha, beta].
    XX, YY : torch.Tensor, shape (n_samples, n_samples)
        Output of the get_grid() function above.
    title : str
    
    Returns
    -------
    fig, ax : plt.figure and axes
    """
    fig, ax = plt.subplots()
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.contour(XX.numpy(), YY.numpy(), p_learned.reshape(1000,1000), cmap='Reds')  # Plot learned distribution
    plt.contour(XX.numpy(), YY.numpy(), p_ana.reshape(1000,1000), cmap='Blues')  # Plot analytic distribution
    plt.plot(theta[0],theta[1], 'ro', color='lightgreen')  # Plot ground truth parameter (dirac)

    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label="Learned"),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label="Analytic"),
        mpatches.Patch(color='lightgreen', label="Ground Truth")]
    plt.legend(handles=handles)
    plt.title(title)
    return fig, ax


def compute_dist_to_dirac(samples, p, theta):
    """Compute the distance between the posterior and the ground-truth dirac measure
    on the grid returned by the function get_grid() above.
    
    Parameters
    ----------
    samples : torch.Tensor, shape (n_samples*n_samples, 2) 
        Output of the get_grid() function above.
    p : ndarray, shape (n_samples)
        Values of the posterior on each point (a,b) of the samples. 
    theta : ndarray, shape (2,)
        Values of the ground-truth parameters [alpha, beta].
    
    Returns
    -------
    dist_to_dirac : float. 
    """
    if theta.ndim == 1:
        theta = theta[None]

    diff = (samples - theta)
    M = (diff * diff).sum(axis=1)
    dist_to_dirac = (M * (p / p.sum())).sum()
    return dist_to_dirac
    
def compute_wasserstein_dist(samples, p_ana, p_learned):
    """Compute the wasserstein distance between the analytic and the learned posterior
    on the grid returned by the function get_grid() above.
    
    Parameters
    ----------
    samples : torch.Tensor, shape (n_samples*n_samples, 2) 
        Output of the get_grid() function above.
    p_ana, p_learned : ndarray, shape (n_samples)
        Output of the get_posteriors_on_grid() function above.
    theta : ndarray, shape (2,)
        Values of the ground-truth parameters [alpha, beta].
    
    Returns
    -------
    W_post : float
        Wasserstein distance (SamplesLoss from geomloss) between p_ana and p_learned.
    """
    try:
            from geomloss import SamplesLoss
            loss = SamplesLoss()
            samples = samples.to(dtype=torch.float64)
            W_post = float(loss(p_ana, samples, p_learned, samples))
    except ImportError:
        W_post = None
        
    return W_post

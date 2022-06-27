"""
Some useful functions to evaluate the learned posterior (flow) by comparing it to 
the analytic posterior and/or the dirac function at the ground-truth parameters. 

These functions enable to compute distances between distributions and plot results.
"""

from statistics import mode
import torch
import numpy as np

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager

def compute_dist_to_dirac(posterior, theta, n_samples=10000):
    """Compute the distance between the posterior and the ground-truth dirac measure.
    
    Parameters
    ----------
    posterior : DirectPosterior object from sbi.inference.posteriors
    theta : list of len 4
        Values of the ground-truth parameters [C, mu, sigma, gain].
    
    Returns
    -------
    dist_to_dirac : float. 
    """
    theta = torch.FloatTensor(theta)

    samples = posterior.sample((n_samples,), sample_with_mcmc=False)

    dist_to_dirac = []
    for j in range(len(theta)):
        samples_coordj = samples[:, j]
        sd = np.sqrt(samples_coordj.var())
        dist_to_dirac.append((samples_coordj.var() + (samples_coordj.mean() - theta[j])**2) / sd)
    
    dist_to_dirac = torch.stack(dist_to_dirac).mean()
    return dist_to_dirac

def plot_pairgrid_with_groundtruth(posteriors, theta_gt, color_dict, handles, n_samples=10000):

    modes = list(posteriors.keys())
    dfs = []
    for n in range(len(posteriors)):
        posterior = posteriors[modes[n]]
        if modes[n]=='prior':
            samples = posterior.sample((n_samples,))
        else:
            samples = posterior.sample((n_samples,), sample_with_mcmc=False)
        df = pd.DataFrame(samples.numpy(),columns=[r'$C$',r'$\mu$',r'$\sigma$',r'$g$'])
        df['mode'] = modes[n]
        dfs.append(df)
    
    joint_df = pd.concat(dfs, ignore_index=True)

    mpl.rcParams["axes.labelsize"] = 25
    mpl.rc('xtick', labelsize=15) 
    mpl.rc('ytick', labelsize=15) 
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    mpl.rcParams['mathtext.fontset'] = 'cm'
   
    g = sns.PairGrid(joint_df, hue='mode', palette=color_dict, diag_sharey=False, corner=True)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, shade=True)

    g.axes[1][0].set_xlim(10.,300.) # C
    g.axes[1][0].set_ylim(50.,500.) # mu
    # g.axes[1][0].set_xticks([])

    g.axes[2][0].set_xlim(10.,300.) # C
    g.axes[2][0].set_ylim(100.,5000.) # sigma
    # g.axes[2][0].set_xticks([])


    g.axes[2][1].set_xlim(50.,500.) # mu
    g.axes[2][1].set_ylim(100.,5000.) # sigma
    # g.axes[2][1].set_xticks([])


    g.axes[3][0].set_xlim(10.,300.) # C
    g.axes[3][0].set_ylim(-20.,20.) # gain
    g.axes[3][0].set_yticks([-20,0,20])

    g.axes[3][1].set_xlim(50.,500.) # mu
    g.axes[3][1].set_ylim(-20.,20.) # gain
    g.axes[3][1].set_xticks([200,400])

    g.axes[3][2].set_xlim(100.,5000.) # sigma
    g.axes[3][2].set_ylim(-20.,20.) # gain
    g.axes[3][2].set_xticks([2000,4000])

    g.axes[3][3].set_xlim(-20.,20.) # gain

    if theta_gt is not None:
        # get groundtruth parameters
        C, mu, sigma, gain = theta_gt

        # plot points
        g.axes[1][0].scatter(C,mu, color='black', zorder=2)
        g.axes[2][0].scatter(C,sigma, color='black', zorder=2)
        g.axes[2][1].scatter(mu,sigma, color='black', zorder=2)
        g.axes[3][0].scatter(C,gain, color='black', zorder=2)
        g.axes[3][1].scatter(mu, gain, color='black', zorder=2)
        g.axes[3][2].scatter(sigma, gain, color='black', zorder=2)
        g.axes[3][3].axvline(x=gain, ls='--', c='black')

        # plot dirac
        g.axes[0][0].axvline(x=C, ls='--', c='black')
        g.axes[1][1].axvline(x=mu, ls='--', c='black')
        g.axes[2][2].axvline(x=sigma, ls='--', c='black')

    font = font_manager.FontProperties(family='serif', size=21)

    plt.legend(handles=handles, prop=font, fontsize=21, title="Extra observations", title_fontsize=20, bbox_to_anchor=(1.1,3.5))
    
    return g 

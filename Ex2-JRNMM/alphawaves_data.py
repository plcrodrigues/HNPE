from alphawaves.dataset import AlphaWaves
import numpy as np
import mne
import torch

import warnings
warnings.filterwarnings("ignore")

def get_alphaeeg_observation(subject_id = 0, tmin=0.0, tmax=8.0, context_event=None):
    ''' Data consists of recordings taken from a public dataset (Cattan et al., 2018) 
    in which subjects were asked to keep their eyes open or closed during periods of 
    8 s (sampling frequency of 128 Hz). For one subject there are ten epochs (5 open eyes 
    events, 5 closed eyes events). We choose the observed signal x_0 to be in the closed 
    eyes state. The other 9 time series define the context. We have used only the recordings 
    from channel Oz because it is placed near the visual cortex and, therefore, is the most 
    relevant channel for the analysis of the open and closed eyes conditions. The signals 
    were filtered between 3 Hz and 40 Hz.
    '''
    # define the dataset instance
    dataset = AlphaWaves(useMontagePosition = False) # use useMontagePosition = False with recent mne versions

    # get the data from subject of interest
    subject = dataset.subject_list[subject_id]
    raw = dataset._get_single_subject_data(subject)

    # filter data and resample
    fmin = 3
    fmax = 40
    raw.filter(fmin, fmax, verbose=False)
    raw.resample(sfreq=128, verbose=False)

    # detect the events and cut the signal into epochs
    events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    event_id = {'closed': 1, 'open': 2}
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=None,
                        verbose=False, preload=True)
    epochs.pick_types(eeg=True)

    # pick the channel of interest
    epochs.pick_channels(ch_names=['Oz'])

    # find first closed event
    i = 0
    while(events[:,2][i] != 1):
        i += 1
    mask = np.array([i==k for k in range(len(events))])

    # define the observed signal (x_0) and context (x_i) epochs 
    x_0 = torch.FloatTensor(epochs[mask].get_data()[:,0,:-1][:,:,None]).permute(2,1,0)  # first closed event 
    if context_event == None:
        observation = x_0
    else:
        context = epochs[~mask]
        if context_event != "all":
            context = context[context_event]
        context = torch.FloatTensor(context.get_data()[:,0,:-1][:,:,None]).permute(2,1,0)  # all other events

        observation = torch.cat([x_0, context], dim=2)

    return observation
    

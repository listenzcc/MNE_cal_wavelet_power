# code:utf-8

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from tools import get_epochs, para_setting


# define MEG parameters setting
file_dir = os.path.join('D:/', 'BeidaShuju', 'rawdata')
fname_list, ortids, event_id, tmin, t0, tmax = para_setting(
    os.path.join(file_dir, 'ZYF'))
freq_h = 80
decim = 1

# define frequencies of interest
freqs = np.linspace(10, 40, num=20)
n_cycles = freqs / 5.  # different number of cycle per frequency
n_jobs = 12

for fname in fname_list:
    print(fname)
    epochs = get_epochs(fname, event_id, tmin, t0, tmax,
                        freq_h=freq_h, decim=decim)
    epochs.average().plot(spatial_colors=True, show=False)

    ec = epochs.copy()
    ec.subtract_evoked()
    ec.average().plot(spatial_colors=True, show=False)

    power, itc = mne.time_frequency.tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles,
        return_itc=True, n_jobs=n_jobs)
    power.plot_joint(fmax=max(freqs), show=False)

    powe_, _ = mne.time_frequency.tfr_morlet(
        ec, freqs=freqs, n_cycles=n_cycles,
        return_itc=True, n_jobs=n_jobs)
    powe_.plot_joint(fmax=max(freqs), show=False)

    plt.show()
    break

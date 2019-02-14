# code:utf-8

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from tools import para_setting, get_epochs, get_tfr_power, save_file


# define MEG parameters setting
file_dir = os.path.join('D:/', 'BeidaShuju', 'rawdata')
subject_name = 'ZYF'
fname_list, ortids, event_id, tmin, t0, tmax = para_setting(
    os.path.join(file_dir, subject_name))
freq_h = 80
decim = 1
dir_save = 'data_storage_'

# define frequencies of interest
freqs = np.linspace(2, 40, num=20)
n_cycles = freqs / 5.  # different number of cycle per frequency
n_jobs = 12

for fname in fname_list:
    print(fname)
    epochs = get_epochs(fname, event_id, tmin, t0, tmax,
                        freq_h=freq_h, decim=decim)
    induces = epochs.copy()
    induces.subtract_evoked()

    power_orts = dict()
    for id in event_id.keys():
        print(id)
        power_orts[id] = get_tfr_power(
            induces[id], freqs=freqs, n_cycles=n_cycles,
            return_itc=False, n_jobs=n_jobs)
        # power_orts[id].plot_joint(fmax=max(freqs), show=False)
    save_file(power_orts, os.path.join(
        dir_save, subject_name+'_'+os.path.basename(fname)))

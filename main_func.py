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

    for id in event_id.keys():
        print(id)
        power_orts_epochs = list(get_tfr_power(
            epochs[id][j], freqs=freqs, n_cycles=n_cycles,
            return_itc=False, n_jobs=n_jobs)
            for j in range(epochs[id].events.shape[0]))
        power_orts_induces = list(get_tfr_power(
            induces[id][j], freqs=freqs, n_cycles=n_cycles,
            return_itc=False, n_jobs=n_jobs)
            for j in range(induces[id].events.shape[0]))

        save_file(power_orts_epochs, os.path.join(dir_save, '_'.join(
            ['epochs', subject_name, os.path.basename(fname), id])))
        save_file(power_orts_induces, os.path.join(dir_save, '_'.join(
            ['induces', subject_name, os.path.basename(fname), id])))

        break
    break

# code: utf-8

import functools
import mne
from mne.time_frequency import tfr_morlet
import os
import pickle
import time


def time_it(fn):

    @functools.wraps(fn)
    def new_fn(*args, **kws):
        print('-' * 60)
        start = time.time()
        result = fn(*args, **kws)
        end = time.time()
        duration = end - start
        print('%s seconds are consumed in executing function:\n\t%s%r' %
              (duration, fn.__name__, args))
        return result
    return new_fn


@time_it
def para_setting(filedir, train=True):
    if train:
        fname_list = list(os.path.join(
            filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
            for j in range(1, 6))
        ortids = [2, 6, 9, 14, 17, 33]
        event_id = dict(ort015=2,  ort045=6,  ort075=9,
                        ort105=14, ort135=17, ort165=33)
        tmin, t0, tmax = -0.2, 0, 0.8
    else:
        fname_list = list(os.path.join(
            filedir, 'MultiTest_%d_raw_tsss.fif' % j)
            for j in range(1, 9))
        ortids = [8, 16, 32, 64]
        event_id = dict(ort45a=8, ort135a=16,
                        ort45b=32, ort135b=64)
        tmin, t0, tmax = -0.4, -0.2, 0.8
    return fname_list, ortids, event_id, tmin, t0, tmax


@time_it
def get_epochs(fname, event_id, tmin, t0, tmax,
               freq_l=1, freq_h=10, decim=1):
    # Make defaults
    baseline = (tmin, t0)
    reject = dict(mag=5e-12, grad=4000e-13)

    # Prepare rawobject
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.filter(freq_l, freq_h, fir_design='firwin')
    picks = mne.pick_types(raw.info, meg=True, eeg=False,
                           eog=False, stim=False, exclude='bads')

    # Get events
    events = mne.find_events(raw)

    # Get epochs
    epochs = mne.Epochs(raw, event_id=event_id, events=events,
                        decim=decim, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=baseline,
                        reject=reject, preload=True)
    return epochs


@time_it
def get_tfr_power(epochs, freqs, n_cycles,
                  return_itc=False, n_jobs=12):
    power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                       return_itc=return_itc, n_jobs=n_jobs)
    return power


@time_it
def save_file(obj, path):
    # pickle can not use 'with open(..) as f'
    # do not know why
    f = open(path+'.pkl', 'wb')
    pickle.dump(obj, f)
    f.close()

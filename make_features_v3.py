#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats

from scipy import ndimage
import numpy as np
import emd
from mne.decoding import SSD    

# Helper function for the second level sift
def mask_sift_second_layer(IA, masks, config={}):
    imf2 = np.zeros((IA.shape[0], IA.shape[1], config['max_imfs']))
    for ii in range(IA.shape[1]):
        config['mask_freqs'] = masks[ii:]
        tmp = emd.sift.mask_sift(IA[:, ii], **config)
        imf2[:, ii, :tmp.shape[1]] = tmp
    return imf2

def get_holo_trl(filename):
    raw=mne.io.read_raw_fif(filename)
    # find alpha
    freqs_sig = 9, 12
    freqs_noise = 8, 13

    epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=False)
    epochs.drop_bad(reject = dict(mag=3e-12))      # unit: T (magnetometers)

    ssd = SSD(info=raw.info,
              reg='oas',
              sort_by_spectral_ratio=False,  # False for purpose of example.
              n_components = 5,
              filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                      l_trans_bandwidth=1, h_trans_bandwidth=1),
              filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                     l_trans_bandwidth=1, h_trans_bandwidth=1))
    ssd.fit(X=epochs.get_data())

    pattern = mne.EvokedArray(data=ssd.patterns_[:4].T,
                              info=ssd.info)
    pattern.plot_topomap(units=dict(mag='A.U.'), time_format='')

    idx = np.argmax(np.abs(ssd.patterns_[0]))
    name_ = raw.info['ch_names'][idx]
    epochs.load_data()
    x = epochs.copy().pick_channels([name_]).get_data()
    x = np.squeeze(x) * 1e+15
    x.shape

    n_trl = epochs.get_data().shape[0]
    sholo_temp = np.zeros((n_trl, 256, 128))

    sample_rate = np.int(raw.info['sfreq'])
    config = emd.sift.get_config('mask_sift')
    config['max_imfs'] = 7
    config['mask_freqs'] = 50/sample_rate
    config['mask_amp_mode'] = 'ratio_sig'
    config['imf_opts/sd_thresh'] = 0.05

    # Carrier frequency histogram definition
    carrier_hist = (1, 100, 256, 'log')
    # AM frequency histogram definition
    am_hist = (1e-2, 32, 128, 'log')

    for n in range(n_trl):
        imf = emd.sift.mask_sift(x[n], **config)
        IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
        masks = np.array([25/2**ii for ii in range(12)])/sample_rate
        config = emd.sift.get_config('mask_sift')
        config['mask_amp_mode'] = 'ratio_sig'
        config['mask_amp'] = 2
        config['max_imfs'] = 5
        config['imf_opts/sd_thresh'] = 0.05
        config['envelope_opts/interp_method'] = 'mono_pchip'

        # Sift the first 5 first level IMFs
        imf2 = emd.sift.mask_sift_second_layer(IA, masks, sift_args=config)

        IP2, IF2, IA2 = emd.spectra.frequency_transform(imf2, sample_rate, 'nht')
        fcarrier, fam, holo = emd.spectra.holospectrum(IF[:,0:6], IF2[:,0:6,:], IA2[:,0:6,:], carrier_hist, am_hist)
        sholo = ndimage.gaussian_filter(holo, 1)

        sholo_temp[n] = sholo
        pass

    sholo_m = np.squeeze(np.nanmean(sholo_temp, axis = 0))
    return sholo_m, fam, fcarrier


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'qt')


# In[ ]:


group = 'control'
path = '/Users/kevinhsu/Documents/D/00_datasets/biomag_2022/holo/train/c/'

for s in range(100):
    try:
        s += 1
        filename = 'de_hokuto_%s%d-raw.fif'%(group, s)

        sholo_m, fam, fcarrier = get_holo_trl(filename)

        fig, ax = plt.subplots()
        ax.pcolormesh(fam, fcarrier, stats.zscore(sholo_m), cmap='hot_r', shading='nearest', vmin = -3, vmax = 3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)



        filename = 'hokuto_%s%d'%(group, s)

        fig_fname = path + filename + '-holo.png'
        fig.savefig(fig_fname, bbox_inches='tight')
    except:
        print('no id %d'%s)
    pass


# In[ ]:





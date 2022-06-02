# -*- coding: utf-8 -*-
# %%
"""
Created on Thu Apr 28 16:12:38 2022

@author: TH
"""
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
method = 'infomax'  #old ver: 'extended-infomax'
n_components = 15
max_pca_components = 15
random_state = 23

decim = 3

picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False,ecg=False, exclude='bads')

ica = ICA(n_components=n_components,  
          random_state=random_state, method=method) #old ver: max_pca_components=max_pca_components,

ica.fit(raw, decim=decim, picks=picks)

ica.plot_components() #ica.plot_components(layout = kit_lay)

ica.plot_sources(raw)

# %%
#check how well it works if certain features are removed.
ica.plot_overlay(raw, exclude=[0,1,3,5,7,8,10,13,14])

# %%
ica.apply(raw, exclude= [0,1,3,5,7,8,10,13,14])
raw.plot()

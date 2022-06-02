#!/usr/bin/env python
# coding: utf-8

# In[83]:


import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
from scipy import stats
from scipy import ndimage
import emd

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Helper function for the second level sift
def mask_sift_second_layer(IA, masks, config={}):
    imf2 = np.zeros((IA.shape[0], IA.shape[1], config['max_imfs']))
    for ii in range(IA.shape[1]):
        config['mask_freqs'] = masks[ii:]
        tmp = emd.sift.mask_sift(IA[:, ii], **config)
        imf2[:, ii, :tmp.shape[1]] = tmp
    return imf2

def get_holo_trl(filename, sys_type):
    raw=mne.io.read_raw_fif(filename)

    if sys_type == 'A':
        ag_num = [13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 46, 48, 56, 57, 58, 60, 61, 62, 63, 64, 74, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 110, 112, 120, 121, 122, 123, 124, 125, 126, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160]

    if sys_type == 'B':
        ag_num = [9, 13, 15, 16, 17, 18, 20, 21, 22, 25, 26, 27, 29, 30, 31, 33, 34, 49, 50, 58, 59, 60, 61, 62, 63, 64, 76, 79, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 110, 114, 121, 123, 124, 125, 126, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160]

    ag_names = ['AG%.3d'%s for s in ag_num]
    ag_picks = mne.pick_channels(raw.ch_names, ag_names)

    # find alpha
    freqs_sig = 9, 12
    freqs_noise = 8, 13

    epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=False)
    epochs.drop_bad(reject = dict(mag=3e-12))      # unit: T (magnetometers)

    psd, freq_ = mne.time_frequency.psd_array_welch(epochs.get_data() * 1e+15, 
                                                    epochs.info['sfreq'], 
                                                    fmin=8, fmax=12, 
                                                    n_fft=256, average='mean')
    psd_M=np.expand_dims(psd[0].mean(axis = 1),axis=1)
    psd_evk=mne.EvokedArray(psd_M,epochs.info,comment='raw_data')
    temp_n = None
    alpha_channels = []
    for i in range(5):
        psd_evk.pick('all',exclude=[temp_n])
        temp_n, temp_t = psd_evk.get_peak()
        alpha_channels.append(temp_n)
        pass

    print(alpha_channels)
    
    epochs.load_data()
    x = epochs.copy().pick_channels([alpha_channels[0]]).get_data()
    x = np.squeeze(x) * 1e+15

    n_trl = epochs.get_data().shape[0]
    sholo_temp = np.zeros((n_trl, 64, 64))

    sample_rate = np.int(raw.info['sfreq'])
    config_lay1 = emd.sift.get_config('mask_sift')
    config_lay1['max_imfs'] = 7
    config_lay1['mask_amp_mode'] = 'ratio_sig'
    config_lay1['imf_opts/sd_thresh'] = 0.05
    config_lay1['verbose'] = 'CRITICAL'

    config_lay2 = emd.sift.get_config('mask_sift')
    config_lay2['mask_amp_mode'] = 'ratio_sig'
    config_lay2['mask_amp'] = 2
    config_lay2['max_imfs'] = 5
    config_lay2['imf_opts/sd_thresh'] = 0.05
    config_lay2['envelope_opts/interp_method'] = 'mono_pchip'
    config_lay2['verbose'] = 'CRITICAL'
    # Carrier frequency histogram definition
    carrier_hist = (1, 100, 64, 'log')
    # AM frequency histogram definition
    am_hist = (1e-2, 32, 64, 'log')

    for n in range(n_trl):
        imf = emd.sift.mask_sift(x[n], **config_lay1)
        IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
        masks = np.array([25/2**ii for ii in range(12)])/sample_rate

        # Sift the first 5 first level IMFs
        imf2 = emd.sift.mask_sift_second_layer(IA, masks, sift_args=config_lay2)

        IP2, IF2, IA2 = emd.spectra.frequency_transform(imf2, sample_rate, 'nht')
        fcarrier, fam, holo = emd.spectra.holospectrum(IF, IF2, IA2, carrier_hist, am_hist)
        sholo = ndimage.gaussian_filter(holo, 1)

        sholo_temp[n] = sholo
        pass

    sholo_m = np.squeeze(np.nanmean(sholo_temp, axis = 0))
    return sholo_m, fam, fcarrier

def holo_gray_images(sholo_m, fam, fcarrier):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(32*px, 32*px))
    fig.subplots_adjust(0,0,1,1)
    ax.pcolormesh(fam, fcarrier, stats.zscore(sholo_m), cmap='hot_r', shading='nearest', vmin = -2, vmax = 2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    im = Image.open(img_buf, )
    imgGray = im.convert('L')
    mage_sequence = imgGray.getdata()
    image_array = np.array(mage_sequence)
    return image_array


# In[39]:


get_ipython().run_line_magic('matplotlib', 'qt')


# In[66]:


group = 'dementia'
path = '/Users/kevinhsu/Documents/D/00_datasets/biomag_2022/holo/train/m/'

s = 10
sys_type = 'A'

filename = 'de_hokuto_%s%d-raw.fif'%(group, s)

raw=mne.io.read_raw_fif(filename)

if sys_type == 'A':
    ag_num = [13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 46, 48, 56, 57, 58, 60, 61, 62, 63, 64, 74, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 110, 112, 120, 121, 122, 123, 124, 125, 126, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160]

if sys_type == 'B':
    ag_num = [9, 13, 15, 16, 17, 18, 20, 21, 22, 25, 26, 27, 29, 30, 31, 33, 34, 49, 50, 58, 59, 60, 61, 62, 63, 64, 76, 79, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 110, 114, 121, 123, 124, 125, 126, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160]

ag_names = ['AG%.3d'%s for s in ag_num]
ag_picks = mne.pick_channels(raw.ch_names, ag_names)

raw.pick(ag_picks)

sholo_m, fam, fcarrier = get_holo_trl(filename, sys_type)


# In[70]:


plt.pcolormesh(fam, fcarrier, sholo_m, cmap='ocean_r', shading='nearest')
plt.title('Holospectrum')
plt.xlabel('AM Frequency (Hz)')
plt.show()
sholo_m.shape


# In[84]:


im_array = holo_gray_images(sholo_m, fam, fcarrier)


# In[87]:


group = 'control'


sys_types = ['A','B','A','A','A','A','A','B','B','B','A',
             'A','A','B','A','A','B','B','B','B','A','A','B','A','A','A','A',
             'A','A','A','B','B','A','A','B','A','A','A','A','B','A','A','A',
             'A','A','A','A','A','B','A','A','A','A','A','A','A','B','A','A',
             'A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A',
             'A','B','B','B','A','A','A','A','A','A','A','A','B','A','B','A',
             'A','A','B','B','B','B','A','B','B']

excluded_control = []

holo_c = []
for i in range(100):
    s = i + 1
    sys_type = sys_types[i]
    try:
        filename = 'de_hokuto_%s%d-raw.fif'%(group, s)
        sholo_m, fam, fcarrier = get_holo_trl(filename, sys_type)
        im_array = holo_gray_images(sholo_m, fam, fcarrier)
        holo_c.append(im_array)
    except:
        print('no id %d'%s)
        excluded_control.append(s)
    pass
    


# In[88]:


group = 'dementia'


sys_types = ['A','B','A','B','A','B','B','B','B',
             'A','B','A','A','A','A','A','A','A',
             'A','B','A','B','A','A','B','A','A',
             'A','A']
excluded_dementia = []
holo_d = []
for i in range(29):
    s = i + 1
    sys_type = sys_types[i]
    try:
        filename = 'de_hokuto_%s%d-raw.fif'%(group, s)
        sholo_m, fam, fcarrier = get_holo_trl(filename, sys_type)
        im_array = holo_gray_images(sholo_m, fam, fcarrier)
        holo_d.append(im_array)
    except:
        print('no id %d'%s)
        excluded_dementia.append(s)
    pass
    


# In[89]:



group = 'mci'


sys_types = ['B','B','B','B','B','B','B','B','B',
             'B','A','B','B','B','B']
excluded_mci = []
holo_m = []
for i in range(15):
    s = i + 1
    sys_type = sys_types[i]
    try:
        filename = 'de_hokuto_%s%d-raw.fif'%(group, s)
        sholo_m, fam, fcarrier = get_holo_trl(filename, sys_type)
        im_array = holo_gray_images(sholo_m, fam, fcarrier)
        holo_m.append(im_array)
    except:
        print('no id %d'%s)
        excluded_mci.append(s)
    pass
    


# In[90]:


print(excluded_control)
print(excluded_dementia)
print(excluded_mci)
holo_2d_d = np.vstack(holo_d)
holo_2d_c = np.vstack(holo_c)
holo_2d_m = np.vstack(holo_m)

holo_2d = np.concatenate((holo_2d_c, holo_2d_d, holo_2d_m),axis=0)

y = [0] * holo_2d_c.shape[0] + [1] * holo_2d_d.shape[0] + [2] * holo_2d_m.shape[0]
y = np.array(y)
print(holo_2d.shape)
print(len(y))


# In[109]:


X = holo_2d / 255.0


wclf = svm.SVC(kernel="linear", class_weight={0: 1, 1: 10, 2:1})
scores = cross_validate(wclf, X, y, cv=10, return_train_score=True)
print(scores['train_score'].mean())
print(scores['test_score'].mean())


# In[110]:




clf = MLPClassifier(solver='adam', learning_rate_init=0.00001, max_iter=20000, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
clf.fit(X_train, y_train)
print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))


# In[96]:


with open('test.npy', 'wb') as f:
    np.save(f, holo_2d)
    np.save(f, holo_2d_c)
    np.save(f, holo_2d_d)
    np.save(f, holo_2d_m)
    np.save(f, y)
#with open('test.npy', 'rb') as f:
#    holo_2d = np.load(f)
#    holo_2d_c = np.load(f)
 #   holo_2d_d = np.load(f)
  #  holo_2d_m = np.load(f)
   # y = np.load(f)


# In[99]:


X / 255.0


# In[ ]:





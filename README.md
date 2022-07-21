# # Analytic Pipeline for BIOMAG DATA COMPETITION (Dementia screening challenge)  

2022.05.30  

Members:  
1Chun-hsien Hsu P.hD. (Kevin Hsu)  
2Chien-Chang Chen P.hD.  
1Tong-hou Cheong. 
1Ting-hsin Yen  
1Yu-chen Wu  
1Wen-yi Lu  
  
1Institute of Cognitive Neuroscience, National Central University, Taiwan  
2Department of Biomedical Science and Engineering, National Central University, Taiwan  
  
Description:  
  
Computational tools are included in the folder ‘toolbox’. The pipeline begins in running some scripts in Matlab, and users should download and install
Fieldtrip toolbox before using this pipline. The denoise_v4 toolbox was downloaded from Prof. Jonathan Simon’s website
(http://cansl.isr.umd.edu/simonlab/Home.html). Matlab codes for EEMD (in two folders: EEMD and FEEMD ) were downloaded from Research Center for Adaptive 
Data Analysis, National Central University, Taiwan, in 2015. The function runica.m is from EEGLAB toolbox. Part of the pipeline is written in Python based 
on the EEG/MEG analysis package mne-python and emd:  
  
https://mne.tools/stable/install/index.html  
https://emd.readthedocs.io/en/stable/index.html  
  
Step 1. (matlab) read_spm2ft.m: read data files and export them into FIF files.  
  
Step 2. (matlab, optional) ICA_EEMD_TSPCA.m: removing noise using TSPCA algorithm. Because the MEG system does not have reference sensors,   
ICA and EEMD algorithms are applied to create synthesized noise.  
  
Step 3. (python) downsample_ICA.py: There is a jupyter notebook file of this step (downsample_ICA.ipynb). Continuous data was   
lowpass filtered at 200 Hz and was resampled to 500 Hz. Some data files were notch-filtered to remove 50 Hz line noise.   
The python script “raw_plot_psd_ecg_eog_template.py” can generate plots of waveforms, power spectral density, ECG activity and   
EOG activity. These figures can visualize the quality of data and help to 1) find ICA components of EOG and ECG and 2) to see   
whether there is 50 Hz line noise. The folder ‘png_esamples’ have examples of figures generated by raw_plot_psd_ecg_eog_template.py.  
  
  
  
Step 4. (python) make_features_psd_topomap_submit_20220530.py: There is a jupyter notebook file of this step. Continuous data was   
segmented into a set of epochs spaced equidistantly in time (2 seconds). Alpha activities (8-12Hz) at each sensors were estimated   
using using Welch’s method. After applying z-transformation, standardized alpha activities were used to create topographic plots.   
To balance the number of samples between groups, each dementia subject’s MEG data was split into four subsets, and each MCI subject’s   
data was split into eight subsets.   
  

fiff_file = 'hokuto_test16-raw.fif';

cfg = []
cfg.dataset = fiff_file;
data1 = ft_preprocessing(cfg);

%
dat = data1.trial{1} * 1e+15;

ICnum = 5;

[weights, sphere, compvars] = runica(dat,  'lrate',   0.001, 'pca', ICnum);
unmixing = weights*sphere;
    
if (size(unmixing,1)==size(unmixing,2)) && rank(unmixing)==size(unmixing,1)
    mixing = inv(unmixing);
else
    mixing = pinv(unmixing);
end

IC_activity = unmixing * dat;

IMFs = [];

for i = 1:5,
    disp(i)
    imfs_ = eemd(IC_activity(i, :), 0.2, 50, 5);
    IMFs = cat(3, IMFs, imfs_);
    disp(size(IMFs))
end

temp_ref = squeeze(IMFs(end, :, :)); % times by IMF
[tsr_data, idx] = tsr(dat', temp_ref, [-100 100]);
%
data1.trial{1} = zeros(size(data1.trial{1}));
data1.trial{1}(:, idx) = tsr_data' * 1e-15;


cfg = [];cfg.viewmode='butterfly';
ft_databrowser(cfg, data1)

%


fiff_file  = 'hokuto_test16-tspca-raw.fif';
fieldtrip2fiff(fiff_file, data1)

%%

comp.fsample = data1.fsample;
comp.time    = data1.time;
comp.trial   = {IC_activity};
comp.topo = mixing;
comp.unmixing = unmixing;
for k = 1:size(comp.topo,2)
  comp.label{k,1} = sprintf('%s%03d', 'IC', k);
end
comp.topolabel = data1.label(:);
%comp.sampleinfo = data.sampleinfo;
comp.compvars = compvars;
comp.cfg = data1.cfg;

cfg = [];cfg.viewmode='butterfly';
ft_databrowser(cfg, comp)

%%
clear comp dat
%%

%%
temp_ref = squeeze(IMFs(end, :, :)); % times by IMF

%%
IMFs_2d = [];
for i = 1:7,
    IMFs_2d = cat(1, IMFs_2d, IMFs(:, :, i));
end

%%
[tsr_data, idx] = tsr(data1.trial{1}', temp_ref, [-100 100]);
%%
data1.trial{1} = zeros(size(data1.trial{1}));
data1.trial{1}(:, idx) = tsr_data';

cfg = [];cfg.viewmode='butterfly';
ft_databrowser(cfg, data1)

%%
tmp_de_ = ft_preproc_denoise(dat, IMFs_2d');

%%
data1.trial{1} = tmp_de_;
cfg = [];cfg.viewmode='butterfly';
ft_databrowser(cfg, data1)

from TDMS import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy import stats
import scipy.io
import pandas as pd
import TFTool
import lsfm_analysis
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
from scipy.stats import multivariate_normal
from sklearn.preprocessing import scale
rng = np.random.RandomState(1337)


if  __name__ == "__main__":
    df = pd.read_csv('patch_list_USBMAC.csv', dtype={'date':str, '#':str})
    ilsfm = df.index[df['type']=='Log sFM']
    fdir = df['path'][35]
    t = Tdms()
    t.loadtdms(fdir, load_sound=False)
    _,para = t.get_stim()
    resp,_ = t.get_dpk()
    resp_r = signal.resample(resp, 500, axis=1)
    resp_z = scipy.stats.zscore(resp_r)
    
    with open('FIR.txt', 'r') as f:
        fir = np.array(f.read().split('\n')[:-1], dtype='float64')
    filt = np.abs(np.fft.fft(fir))
        
    cwt = scipy.io.loadmat('/Volumes/BASASLO/in-vivo_patch/cwt_sound/20210617_001_cwt.mat')
    f = cwt['f']
    wt = cwt['wt'].T[:,0]
    wt_a = []
    for w in wt:
        wt_a.append(w)
    wt_a = np.array(wt_a)
    wt_mean = wt_a.mean(axis=(0,2))
    
    t_for,t_lag = 0.1,0.4
    fs = 250
    wt_p = np.pad(wt_a, [(0,0), (0,0), (int(t_lag*fs),int(t_for*fs))], 'constant')
    
    epochs = len(wt_p)
    n_cwt = len(f)
    coeff = []
    for epoch in range(epochs):
        _cwt_coef = []
        for i in range(n_cwt):
            _corr = np.correlate(wt_p[epoch,i,:],resp_r[epoch,:],mode='valid')
            _cwt_coef.append(_corr)
        coeff.append(_cwt_coef)
    
    coeff = np.array(coeff)
    strf = np.mean(coeff, axis=0)
    delays_samp = np.arange(np.round(t_for * -1*fs),
                        np.round(t_lag * fs) + 1).astype(int)
    delays_sec = -1* delays_samp[::-1] / fs
    plt.pcolormesh(delays_sec, f, strf, shading='nearest')
    plt.yscale('log')
    plt.ylim(2000,100000)
    plt.show()
    
    
def cov(arr):
    return np.convolve(arr, rfilt, mode='same')
    
wt_con = np.apply_along_axis(cov, 2, wt_2)

# =============================================================================
# strf = scipy.io.loadmat('/Users/POW/Desktop/STRF/strf_out_0730_002.mat')
# cwt = scipy.io.loadmat('/Users/POW/Documents/Python_Project/lsfm_analysis/20210730_002_cwt.mat' 
# )
# feature = strf['feature']
# coef = strf['coef']
# freqs = feature.T[:,0]
# 
# tmin, tmax = -0.1, 0.4
# sfreq = 250
# delays_samp = np.arange(np.round(tmin * sfreq),
#                         np.round(tmax * sfreq) + 1).astype(int)
# delays_sec = delays_samp / sfreq
# grid = np.array(np.meshgrid(delays_sec, freqs))
# grid = grid.swapaxes(0, -1).swapaxes(0, 1)
# 
# # Simulate a temporal receptive field with a Gabor filter
# means_high = [.05, 6000]
# means_low = [.1, 9000]
# cov = [[.0006, 0], [0, 400000]]
# gauss_high = multivariate_normal.pdf(grid, means_high, cov)
# gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
# weights = gauss_high + gauss_low  # Combine to create the "true" STRF
# kwargs = dict(vmax=np.abs(weights).max(), vmin=-np.abs(weights).max(),
#               cmap='RdBu_r', shading='gouraud')
# 
# fig, ax = plt.subplots()
# ax.pcolormesh(delays_sec, freqs, weights, **kwargs)
# ax.set(title='Simulated STRF', xlabel='Time Lags (s)', ylabel='Frequency (Hz)')
# plt.setp(ax.get_xticklabels(), rotation=45)
# plt.autoscale(tight=True)
# mne.viz.tight_layout()
# 
# 
# # Reshape audio to split into epochs, then make epochs the first dimension.
# n_epochs, n_seconds = 100, 2
# wt = []
# for x in range(300,400):
#     wt.append(cwt['wt'][0][:][x][:])
#     
# X = np.array(wt)**2
# n_times = X.shape[-1]
# 
# # Delay the spectrogram according to delays so it can be combined w/ the STRF
# # Lags will now be in axis 1, then we reshape to vectorize
# delays = np.arange(np.round(tmin * sfreq),
#                    np.round(tmax * sfreq) + 1).astype(int)
# 
# # Iterate through indices and append
# X_del = np.zeros((len(delays),) + X.shape)
# for ii, ix_delay in enumerate(delays):
#     # These arrays will take/put particular indices in the data
#     take = [slice(None)] * X.ndim
#     put = [slice(None)] * X.ndim
#     if ix_delay > 0:
#         take[-1] = slice(None, -ix_delay)
#         put[-1] = slice(ix_delay, None)
#     elif ix_delay < 0:
#         take[-1] = slice(-ix_delay, None)
#         put[-1] = slice(None, ix_delay)
#     X_del[ii][tuple(put)] = X[tuple(take)]
# 
# # Now set the delayed axis to the 2nd dimension
# X_del = np.rollaxis(X_del, 0, 3)
# X_del = X_del.reshape([n_epochs, -1, n_times])
# n_features = X_del.shape[1]
# weights_sim = weights.ravel()
# 
# # Simulate a neural response to the sound, given this STRF
# y = np.zeros((n_epochs, n_times))
# for ii, iep in enumerate(X_del):
#     # Simulate this epoch and add random noise
#     noise_amp = .002
#     y[ii] = np.dot(weights_sim, iep) + noise_amp * rng.randn(n_times)
# 
# # Plot the first 2 trials of audio and the simulated electrode activity
# X_plt = scale(np.hstack(X[:2]).T).T
# y_plt = scale(np.hstack(y[:2]))
# time = np.arange(X_plt.shape[-1]) / sfreq
# _, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
# ax1.pcolormesh(time, freqs, X_plt, vmin=0, vmax=4, cmap='Reds',
#                shading='gouraud')
# ax1.set_title('Input auditory features')
# ax1.set(ylim=[freqs.min(), freqs.max()], ylabel='Frequency (Hz)')
# ax2.plot(time, y_plt)
# ax2.set(xlim=[time.min(), time.max()], title='Simulated response',
#         xlabel='Time (s)', ylabel='Activity (a.u.)')
# mne.viz.tight_layout()
# =============================================================================



"""
df = pd.read_csv('patch_list_USBMAC.csv', dtype={'date':str, '#':str})
ilsfm = df.index[df['type']=='Pure Tones']
fdir = df['path'][29]
t = Tdms()
t.loadtdms(fdir, protocol=1, precise_timing=True)
stim,para = t.get_stim()
resp,_ = t.get_dpk()
sound = t.get_sound()
"""

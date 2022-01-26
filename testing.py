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
import lsfm

if  __name__ == "__main__":
    df = pd.read_csv('patch_list_Q.csv', dtype={'date':str, '#':str})
    idx_lsfm = df.index[df['type']=='Log sFM']
    
    df_loc = 35
    fdir = df['path'][df_loc]
    filename = df['date'][df_loc]+'_'+df['#'][df_loc]
    t = Tdms()
    t.loadtdms(fdir, load_sound=True)
    _,para = t.get_stim()
    resp,_ = t.get_dpk()
    sound,_ = t.get_raw()
    
    with open('FIR_07_27_2021.txt', 'r') as file:
        fir = np.array(file.read().split('\n')[:-1], dtype='float64')
    _fir_fft = np.fft.fft(fir)
# =============================================================================
#     dc = _fir_fft[0]
#     _fir_fft[-19:] = _fir_fft[-20]
#     fir_R = _fir_fft[513:]
#     fir_R = np.around(fir_R, decimals = 12)
#     fir_L = np.conj(np.flip(fir_R))
#     fir_flip = np.hstack((fir_L, fir_R))
#     theta = np.angle(fir_flip, deg=False)
#     filt = np.abs(fir_flip)
#     r = filt[len(filt)//2]/filt
#     inv_filt = r*np.cos(theta) + r*np.sin(theta)*1j
#     inv_filt = np.hstack((0, inv_filt))
#     inv_ifilt = np.fft.ifft(inv_filt)
#     sound_re = np.convolve(sound, np.real(inv_ifilt), 'same')
#     stim = t.cut(sound_re)
#     scipy.io.savemat('4cwt_fir_flip_round_real.mat', {'stim':stim})
# =============================================================================
    
# =============================================================================
#     cf,bd,mod,_ = zip(*para)
#     i=422
#     plt.plot(stim[i])
#     plt.plot(resp[i])
#     ax = plt.subplot()
#     txt = f'{i}, fc:{cf[i]}, Bw:{bd[i]}, Mod:{mod[i]}'
#     ax.text(0.02, 1.03, txt, transform=ax.transAxes, fontsize=12,
#             horizontalalignment='left')
#     plt.savefig(f'stim-resp_{filename}_{i}', dpi=500)
# =============================================================================
    
# =============================================================================
#     #cwt = scipy.io.loadmat(r'E:\Documents\PythonCoding\test_invFIR\fir_theta0pi.mat')
#     cwt = scipy.io.loadmat('cwt_fir_flip_0dc.mat')
#     #resp_r = signal.resample(resp, 500, axis=1)
#     #resp_z = stats.zscore(resp_r)
#     f = cwt['f']
#     f = f[:,0]
#     wt = cwt['wt'].T[:,0]
#     wt_a = []
#     for w in wt:
#         wt_a.append(w)
#     wt_a = np.array(wt_a)
#     wt_mean = wt_a.mean(axis=(0,2))
#     plt.plot(wt_mean)
#     plt.savefig('fir_flip_real.png', dpi=500)
# =============================================================================
    
    
# =============================================================================
#     #plotting resp versus CWT stimulus
#     r = signal.resample(resp[157], 500)
#     w = wt[157]
#     fig = plt.figure()
#     fig.tight_layout()
#     ax1 = plt.subplot2grid(shape=(3, 1), loc=(0, 0), rowspan=2)
#     ax2 = plt.subplot2grid(shape=(3, 1), loc=(2, 0), rowspan=1)
#     plt.subplots_adjust(hspace=0.3)
#     x = np.linspace(0,500,500)
#     ax1.pcolormesh(x, f, w, shading='nearest')
#     ax1.set_ylim(2000,10000)
#     ax1.set_xlim(0,500)
#     ax1.set_yscale('log')
#     ax2.plot(r)
#     ax2.set_xlim(0,500)
#     ax2.text(0.02, 1.03, f'{para[157]}', transform=ax1.transAxes, fontsize=13,
#         horizontalalignment='left')
#     plt.savefig('powerspec_analysis.png', dpi=500)
# =============================================================================
    
    

    


    
#wt_con = np.apply_along_axis(lambda x: np.convolve(x, rfilt, mode='same'), 2, wt_p)

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


from TDMS_ver1 import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy import stats
import scipy.io
import mne
import TFTool
from mne.decoding import ReceptiveField, TimeDelayingRidge
import pandas as pd
import lsfm

def find_slope(spectrum, peak_loc):
    #previous point (higher frequency)
# =============================================================================
#     if len(p1) == len(p2) == len(peaks):
#         slope_pre =  (f[i_freq-1] - f[i_freq]) / ((p1[i_peak] - peaks[i_peak])*4)
#         slope_post = (f[i_freq+1] - f[i_freq]) / ((p2[i_peak] - peaks[i_peak])*-4)
#         return (slope_pre + slope_post)/2
# =============================================================================
    
    spectrum = wt_s[250]
    max_pow = []
    spec = np.swapaxes(spectrum,0,1)
    for i in spec:
        max_pow.append(f[np.argmax(i)])
    
    slope1 = (max_pow[peak_loc] - max_pow[peak_loc-1])/(peak_loc - (peak_loc-1))
    slope2 = (max_pow[peak_loc+1] - max_pow[peak_loc])/((peak_loc + 1) - peak_loc)
    return (slope1+slope2)/2/250     
        
            
            
    
    

if  __name__ == "__main__":
    df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
    idx_lsfm = df.index[df['type']=='Log sFM']
    
    df_loc = 28
    fdir = df['path'][df_loc]
    filename = df['date'][df_loc]+'_'+df['#'][df_loc]
    t = Tdms()
    t.loadtdms(fdir, load_sound=True)
    
    _,para = t.get_stim()
    resp,_ = t.get_dpk()
    sound,_ = t.get_raw()
   
    cwt = scipy.io.loadmat(r'E:\in-vivo _patch_analysis\cwt_fir_real.mat')
    resp_r = signal.resample(resp, 500, axis=1)
    #resp_z = stats.zscore(resp_r)
    f = cwt['f']
    f = f[:,0]
    wt = cwt['wt'].T[:,0]
    wt_a = []
    for w in wt:
        wt_a.append(w)
    wt_a = np.array(wt_a)
    
    
    _, _, mod, _ = zip(*para)
    #use mod_rate at 1.0, 2.0, 8.0, 16.0 to avoid response contamination
    slow = [i for i, a in enumerate(mod) if a >=1.0 and a <= 16.0]
    para_s, wt_s, resp_s = [],[],[]
    for i in slow:
        para_s.append(para[i])
        wt_s.append(wt[i])
        resp_s.append(resp[i])
    
    resp_s = np.array(resp_s)
    
    target_freq = [3,6,12,24,36,48,60,72]
    target_freq = [i*1000 for i in target_freq]
    
    for freq in target_freq:
        i_freq = TFTool.find_nearest(freq, f)       
        peak_store = []
        peak_pre, peak_post = [],[]
        windows = []
        slopes = []
        for idx,spectrum in enumerate(wt_s):
            peaks,peak_paras = signal.find_peaks(spectrum[i_freq], prominence=0.2)
            p1,_ = signal.find_peaks(spectrum[i_freq-1], prominence=0.2)
            p2,_ = signal.find_peaks(spectrum[i_freq+1], prominence=0.2)
            peak_store.append(peaks*100)
            peak_pre.append(p1*100)
            peak_post.append(p2*100)


            if len(peaks) != 0:
                for i,x in enumerate(peaks):
                    slopes.append(find_slope(spectrum, x))
                    x = x*100
                    windows.append(resp_s[idx][x-1250:x+3750])
                    
                
        windows_mean = np.mean(windows, axis=0)
        plt.plot(np.mean(windows, axis=0))
        plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
        ax = plt.subplot()
        txt = (f'{freq} Hz. Averaged from {len(windows)}')
        ax.text(0,1.02, txt, horizontalalignment='left', transform=ax.transAxes)
        plt.show()
    

    
    
# =============================================================================
#     wt_swap = np.swapaxes(wt_a, 0,1)
#     fband = []
#     for w in wt_swap:
#         fband.append(w.max())
#         
#     plt.plot(fband)
#     #plt.savefig('fir_flip_real.png', dpi=500)
# =============================================================================


# =============================================================================
#     with open('FIR_07_27_2021.txt', 'r') as file:
#         fir = np.array(file.read().split('\n')[:-1], dtype='float64')
#     sound, _ = t.get_raw()
#     sound_re = lsfm.inv_fir(sound, fir)
#     sound_re = t.cut(sound_re)
#     scipy.io.savemat(f'{filename}_invfir4cwt.mat', {'stim':sound_re})
#     
#     cwt = scipy.io.loadmat('0730_fir_cwt.mat')
# =============================================================================

    
    
# =============================================================================
# cwt = scipy.io.loadmat('/Users/POW/Desktop/python_learning/cwt_sound.mat')
# f = cwt['f']
# 
# n_epochs = len(resp)
# wt = []
# R = []
# for x in range(n_epochs):
#     R.append(resp[x])
#     wt.append(cwt['wt'][0][:][x][:])
#     
# #cwt_f = np.array(y.T)[:][0]
# 
# R = np.array(R)
# wt = np.array(wt)
# R = signal.resample(R, 500, axis=1)
# P = wt**2
# 
# tmin = -0.1
# tmax = 0.4
# sfreq = 250
# freqs = f.T[:][0]
# 
# train, test = np.arange(n_epochs - 1), n_epochs - 1
# X_train, X_test, y_train, y_test = P[train], P[test], R[train], R[test]
# X_train, X_test, y_train, y_test = [np.rollaxis(ii, -1, 0) for ii in
#                                     (X_train, X_test, y_train, y_test)]
# # Model the simulated data as a function of the spectrogram input
# alphas = np.logspace(-3, 3, 7)
# scores = np.zeros_like(alphas)
# models = []
# for ii, alpha in enumerate(alphas):
#     rf = ReceptiveField(tmin, tmax, sfreq, freqs, estimator=alpha)
#     rf.fit(X_train, y_train)
# 
#     # Now make predictions about the model output, given input stimuli.
#     scores[ii] = rf.score(X_test, y_test)
#     models.append(rf)
# 
# times = rf.delays_ / float(rf.sfreq)
# 
# # Choose the model that performed best on the held out data
# ix_best_alpha = np.argmax(scores)
# best_mod = models[ix_best_alpha]
# coefs = best_mod.coef_[0]
# best_pred = best_mod.predict(X_test)[:, 0]
# 
# # Plot the original STRF, and the one that we recovered with modeling.
# 
# plt.pcolormesh(times, rf.feature_names, coefs, shading='auto')
# #plt.set_title('Best Reconstructed STRF')
# #plt.autoscale(tight=True)
# strf_o = {'time' : times, 'feature' : rf.feature_names, 
#           'coef' : coefs}
# plt.yscale('log')
# plt.ylim(2000,90000)
# plt.savefig('strf.png', dpi=300)
# scipy.io.savemat('strf_out.mat', strf_o)
# =============================================================================





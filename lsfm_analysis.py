from TDMS_ver1 import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy import stats
from scipy import interpolate
import scipy.io
import mne
import TFTool
from mne.decoding import ReceptiveField, TimeDelayingRidge
import pandas as pd
import lsfm 
import math

def stim4contour(stim, para):
    def ranges(arr):
        arr_crop = arr[5000:35000]
        for idx, val in enumerate(arr):
            if val < min(arr_crop) or val > max(arr_crop):
                arr[idx] = 0
                
        return arr
    
    fs = 200000
    """cwt decimation rate is 800 to 250Hz"""
    b,a = signal.butter(3, 150, btype='low', fs=fs)
    hil = signal.hilbert(stim)
    phase = np.unwrap(np.angle(hil))
    ifreq = np.diff(phase) / (2*np.pi) * fs
    filt_ifreq = signal.filtfilt(b,a,ifreq)
    
    stim_lowres = signal.resample(stim, len(stim)/8)
    ifreq_lowres = signal.resample(filt_ifreq, len(filt_ifreq)/8)
    
    
    
if  __name__ == "__main__":
    df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
    idx_lsfm = df.index[df['type']=='Log sFM']
    #tlsfm = [23,24,25,27,28,30,32,35,37,40,45,49]
    #psth_para = pd.DataFrame(columns = ['name','sum','max','min','average','zmax','zmin',
    #                                    'sum1','sum2','sum3','sum4', 'sum5']) 
    df_loc = 28
    if df_loc == 28:

        fdir = df['path'][df_loc]
        filename = df['date'][df_loc]+'_'+df['#'][df_loc]
        t = Tdms()
        t.loadtdms(fdir, load_sound=True, precise_timing=True)
        
        para = t.Para
        resp = np.array(t.Rdpk)
        sound = t.rawS
        stim = t.Sound
           
        
    
# =============================================================================
#         _para = np.swapaxes(np.array(para),0,1)
#         mod_label = sorted(set(_para[2][:]))
#         cf_label = sorted(set(_para[0][:]))
#         bw_label = sorted(set(_para[1][:]))
# =============================================================================
        
# =============================================================================
#         p = lsfm.Psth(resp, para, filename)
#         _ = p.psth_para(plot=False)
#         p.psth_trend(window=(3000,5000))
#         p.psth_all()
#         p.psth_window(((3000,5000)), 'onset')
# =============================================================================
        
# =============================================================================
#         p.psth_window((1250,2750), 'inhibit', saveplot=True, savenotes=False)
#         p.psth_window((5000,10000), 'onset', saveplot=True, savenotes=False)
#         p.psth_window((27500,37500), 'sustain', saveplot=True, savenotes=False)
#         p.psth_window((40000,45000), 'offset', saveplot=True, savenotes=True)
# =============================================================================
        
                                         
            
            
# =============================================================================
#         psth = lsfm.psth(resp, filename)
#         zpsth = stats.zscore(psth)
#         
#         psth_para.loc[idx] = filename, sum(psth), max(psth), min(psth), np.mean(psth),\
#             max(zpsth), min(zpsth), sum(psth[:10000]), sum(psth[10000:20000]), sum(psth[20000:30000]),\
#                 sum(psth[30000:40000]), sum(psth[40000:50000])
#         
#         print(idx)
# =============================================================================
                    
# =============================================================================
#     psth_para.to_csv('PSTH_parameters.csv')
#     types = ['two peaks', 'two peaks', 'two peaks', 'no response', 'two peaks', 'two peaks', 
#              'no response', 'plateau', 'plateau', 'no response', 'plateau', 'no response']    
# =============================================================================
        
    
    
            
            
# =============================================================================
#         plt.plot(test)
#         plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
#         plt.axvline(x=38750, color='k', linestyle='--', alpha=0.5)
#         label = list(np.round(np.linspace(0, 2.0, 11), 2))
#         plt.xticks(np.linspace(0,50000,11),label)
#         ax = plt.subplot()
#         txt = (f'{filename}-PSTH')
#         ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)        
# =============================================================================
           
           
# =============================================================================
#         """response at target frequency"""
#         cwt = scipy.io.loadmat(r'E:\In-Vivo_Patch_Results\FIR\cwt_fir_real.mat')
#         #cwt = scipy.io.loadmat(r'R:\In-Vivo_Patch_Results\FIR\cwt_fir_real.mat')
#         atf = lsfm.RespAtFreq()
#         atf.mod_reduction(stim, resp, para, df, df_loc, cwt)
#         atf.resp_at_freq(nth_freq=True, plot=False)
# =============================================================================
    
        
 
        
# =============================================================================
#         """ average response in positive vs negative slope"""
#         for tf in range(len(atf.target_freq)):   
#             post, neg = [],[]
#             for i,a in enumerate(atf.slopes[tf]):
#                 if a > 0:
#                     post.append(atf.windows[tf][i])
#                 else:
#                     neg.append(atf.windows[tf][i])
#             
#             mean_p = np.mean(post, axis=0)
#             mean_p = mean_p - mean_p[1250]
#             mean_n = np.mean(neg, axis=0)
#             mean_n = mean_n - mean_n[1250]
#             
#             plt.plot(mean_p)
#             plt.plot(mean_n)
#             plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
#             ax = plt.subplot()
#             txt = (f'{atf.filename} - {atf.target_freq[tf]} Hz.')
#             ax.text(0,1.02, txt, horizontalalignment='left', transform=ax.transAxes)    
#             #plt.savefig(f'slope_PvN-{atf.filename}-{atf.target_freq[tf]}.png', dpi=500)
#             plt.show()
#             plt.clf()
# =============================================================================

    
    
    
# =============================================================================
#     resp_r = signal.resample(resp, 500, axis=1)
#     #resp_z = stats.zscore(resp_r)
#     f = cwt['f']
#     f = f[:,0]
#     wt = cwt['wt'].T[:,0]
#     wt_a = []
#     for w in wt:
#         wt_a.append(w)
#     wt_a = np.array(wt_a)
#     
#     _, _, mod, _ = zip(*para)
#     #use mod_rate at 1.0, 2.0, 8.0, 16.0 to avoid response contamination
#     slow = [i for i, a in enumerate(mod) if a >=1.0 and a <= 16.0]
#     para_s, wt_s, resp_s, stim_s = [],[],[],[]
#     for i in slow:
#         para_s.append(para[i])
#         wt_s.append(wt[i])
#         resp_s.append(resp[i])
#         stim_s.append(stim[i])
#     
#     resp_s = np.array(resp_s)
#     
#     
#     #lsfm.nth_resp(resp,para,cwt)
#     
#     fs = 200000
#     
#     inst_freqs = []
#     b,a = signal.butter(2, 6000, btype='low', fs=fs)
#     for stim in stim_s:
#         h = signal.hilbert(stim)
#         phase = np.unwrap(np.angle(h))
#         insfreq = np.diff(phase) / (2*np.pi) * fs
#         insfreq = signal.filtfilt(b,a,insfreq)
#         inst_freqs.append(np.diff(insfreq))
# =============================================================================
        
    
   
    
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





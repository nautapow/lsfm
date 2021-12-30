from TDMS import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import scipy.io
import TFTool
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
import pandas as pd

def mem_V(stim, para, resp, filename=''):
    on_r, off_r = [],[]
    sum_on, sum_off = [],[]
    on_p, on_m, off_p, off_m = [],[],[],[]
    
    """use the sum of PSP amplitude to plot character frequency
    #use 20ms at begining to get baseline for substraction
    #on = from onset of sound stimulus to 94ms later
    #off = offset of sound with 100ms duration
    """
    for i in range(len(resp)):
        base = np.mean(resp[i][:500])
        on_r.append(resp[i][500:3000]-base)
        off_r.append(resp[i][3000:5500]-base)
        sum_on.append(sum(on_r[i]))
        sum_off.append(sum(off_r[i]))
        
        
        if (sum(on_r[i]) >= 0):
            on_p.append(i)
        else:
            on_m.append(i)
        
        if (sum(off_r[i]) >= 0):
            off_p.append(i)
        else:
            off_m.append(i)
    
    #plot charcter frequency
    #for membrane potential at stimulus onset
    on_p = np.array(on_p, dtype='int')
    on_m = np.array(on_m, dtype='int')
    off_p = np.array(off_p, dtype='int')
    off_m = np.array(off_m, dtype='int')
    
    loud, freq, _ = zip(*para)
    freq = np.array(freq)
    loud = np.array(loud)
    sum_on = np.array(sum_on)
    _sum = 300*sum_on/max(np.abs(sum_on))
    
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    ax.set_xscale('log')
    sca1 = ax.scatter(freq[on_p], loud[on_p], s=_sum[on_p], 
                c=_sum[on_p],cmap = 'Reds')
    #plt.colorbar(sca1)
    sca2 = ax.scatter(freq[on_m], loud[on_m], s=np.abs(_sum[on_m]),
                c=_sum[on_m], cmap = 'Blues_r')
    #plt.colorbar(sca2)
    ax.text(0.05, 1.02, filename, fontsize=16, transform=ax.transAxes)
    ax.text(0.3, 1.02, 'On', fontsize=16, transform=ax.transAxes)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Frequency Hz', fontsize=16)
    plt.ylabel('Loudness dB-SPL', fontsize=16)
    plt.savefig(f'{filename}_on', dpi=500)
    #plt.show()
    plt.clf()
    
    
    #for membrane potential at stimulus offset
    sum_off = np.array(sum_off)
    _sum = 300*sum_off/max(np.abs(sum_off))  
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    ax.set_xscale('log')
    sca1 = ax.scatter(freq[off_p], loud[off_p], s=_sum[off_p], 
                c=_sum[off_p],cmap = 'Reds')
    #plt.colorbar(sca1)
    sca2 = ax.scatter(freq[off_m], loud[off_m], s=np.abs(_sum[off_m]),
                c=_sum[off_m], cmap = 'Blues_r')
    #plt.colorbar(sca2)
    ax.text(0.05, 1.02, filename, fontsize=16, transform=ax.transAxes)
    ax.text(0.3, 1.02, 'Off', fontsize=16, transform=ax.transAxes)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Frequency Hz', fontsize=16)
    plt.ylabel('Loudness dB-SPL', fontsize=16)
    plt.savefig(f'{filename}_off', dpi=500)
    #plt.show()
    plt.clf()
    

def avg_freq(stim, para, resp):
    # delet 3200Hz to match averaging every 5 frequencies
    idx = np.arange(0,357,51)
    
    _r = np.array(resp[:])
    _r = np.delete(_r,idx,axis=0)
    resp_avg = _r.reshape(-1,5,10000).mean(axis=1)
    
    _p = np.array(para[:])
    _p = np.delete(_p,idx,axis=0)
    para_avg = _p.reshape(-1,5,3).mean(axis=1, dtype=np.int32)
    
    _s = np.array(stim[:])
    _s = np.delete(_s,idx,axis=0)
    stim_avg = _s.reshape(-1,5,10000).mean(axis=1)
    
    return stim_avg, para_avg, resp_avg
# =============================================================================
# def avg_freq(stim, para, resp):
#     _r = resp[:]
#     #del _r[::71]
#     _r = np.array(_r)
#     resp_avg = np.mean(_r.reshape(-1,7,10000), axis = 1)
#     
#     _p = para[:]
#     #del _p[::71]
#     _p = np.array(_p)
#     para_avg = np.mean(_p.reshape(-1,7,3), axis = 1, dtype=np.int32)
#     
#     _s = stim[:]
#     #del _s[::71]
#     _s = np.array(_s)
#     stim_avg = np.mean(_s.reshape(-1,7,10000), axis = 1)
#     
#     return stim_avg, para_avg, resp_avg
# =============================================================================


def plot_avg_resp(stim, para, resp, filename=''):
    """ALIGN AVERAGE RESPONSE FROM DIFFERENT LOUDNESS"""
    #mem_V(stim, para, resp)
    #mem_V(*avg_freq(stim, para, resp))
    _,_para_avg,_resp_avg = avg_freq(stim, para, resp)
    
    """plot average response of same frequency"""
    fig = plt.figure()
    ax1 = plt.subplot()
    legend = str(range(30,100,10))
    x = np.linspace(0,10000,6)
    xticks = np.linspace(0,400,6, dtype='int')
    for i in range(10):
        for j in range(i,70,10):
            plt.plot(_resp_avg[j]-_resp_avg[j][0], label = '%s'
                     %_para_avg[j][0])
            
        plt.legend()
        ax2 = plt.subplot()
        ax2.text(0.1,1.02,f'{filename}_{_para_avg[j][1]} Hz', transform=ax2.transAxes, fontsize=14)
        plt.xlabel('Response (ms)', fontsize=12)
        plt.xticks(x,xticks)
        plt.savefig(f'{filename}_{i}.png', dpi=500)
        #plt.show()
        plt.clf()
        
        
def sound4strf(para, resp, sound):
    loud,_,_ = zip(*para)
    index = [i for i, a in enumerate(loud) if a==80]
    sound_80 = sound[min(index):max(index)]
    resp_80 = resp[min(index):max(index)]
    return resp_80, sound_80
    

"""     
    #for counting the total number of dB and frequency used
    loud, freq, _ = zip(*para)
    n_loud = [[x, loud.count(x)] for x in set(loud)]
    n_loud.sort()
    _n = [x[1] for x in n_loud]
    _n = max(_n)
"""



if  __name__ == "__main__":
    df = pd.read_csv('patch_list_Q.csv', dtype = {'date':str, '#':str})
    index = df.index[df['type']=='Pure Tones']
    #for i in index:
    i=26
    if i == 26:
        path = df['path'][i]
        filename = df['date'][i]+'_'+df['#'][i]
        try:
            t = Tdms()
            t.loadtdms(path, protocol=1, dePeak=False, precise_timing=True)
            stim,para = t.get_stim()
            resp,_ = t.get_dpk()
            
            #mem_V(stim, para, resp, filename)
            #plot_avg_resp(stim, para, resp, filename)
            #mem_V(*avg_freq(stim, para, resp), filename)
        except:
            pass
    
# =============================================================================
#     resp_80, sound_80 = sound4strf(para, resp, sound)
#     
#     cwt = scipy.io.loadmat(r'E:\Documents\PythonCoding\sound80.mat')
#     f = cwt['f']
# 
#     n_epochs = len(resp_80)
#     wt = []
#     R = []
#     for x in range(n_epochs):
#         R.append(resp_80[x])
#         wt.append(cwt['wt'][0][:][x][:])
#     
#     R = np.array(R)
#     wt = np.array(wt)
#     R = signal.resample(R, 100, axis=1)
#     P = wt**2
#         
#     tmin = 0
#     tmax = 0.25
#     sfreq = 250
#     freqs = f.T[:][0]
# 
#     train, test = np.arange(n_epochs - 1), n_epochs - 1
#     X_train, X_test, y_train, y_test = P[train], P[test], R[train], R[test]
#     X_train, X_test, y_train, y_test = [np.rollaxis(ii, -1, 0) for ii in
#                                         (X_train, X_test, y_train, y_test)]
#     # Model the simulated data as a function of the spectrogram input
#     alphas = np.logspace(-3, 3, 7)
#     scores = np.zeros_like(alphas)
#     models = []
#     for ii, alpha in enumerate(alphas):
#         rf = ReceptiveField(tmin, tmax, sfreq, freqs, estimator=alpha)
#         rf.fit(X_train, y_train)
# 
#         # Now make predictions about the model output, given input stimuli.
#         scores[ii] = rf.score(X_test, y_test)
#         models.append(rf)
# 
#     times = rf.delays_ / float(rf.sfreq)
# 
#     # Choose the model that performed best on the held out data
#     ix_best_alpha = np.argmax(scores)
#     best_mod = models[ix_best_alpha]
#     coefs = best_mod.coef_[0]
#     best_pred = best_mod.predict(X_test)[:, 0]
# 
#     # Plot the original STRF, and the one that we recovered with modeling.
# 
#     plt.pcolormesh(times, rf.feature_names, coefs, shading='auto')
#     #plt.set_title('Best Reconstructed STRF')
#     #plt.autoscale(tight=True)
#     strf_o = {'time' : times, 'feature' : rf.feature_names, 
#               'coef' : coefs}
#     plt.yscale('log')
#     plt.ylim(1000,100000)
#     plt.savefig('strf.png', dpi=300)
#     #scipy.io.savemat('strf_out.mat', strf_o)
# =============================================================================
    


    
    
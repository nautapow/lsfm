import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.image import NonUniformImage
from pathlib import Path
from scipy import signal
from scipy import stats
from scipy.optimize import curve_fit
from scipy import ndimage
import TFTool
import mne
import math
from mne.decoding import ReceptiveField, TimeDelayingRidge
import pandas as pd

def best_freq(resp_tune, para):
    _, freq, _ = zip(*para)
    freq = sorted(set(freq))
    mask = resp_tune[2:] >0
    bf_loc = ndimage.center_of_mass(resp_tune[2:], labels=mask)[1]
    bf = np.interp(bf_loc, np.arange(0,len(freq)), freq)
    
    return bf
    
# =============================================================================
#     def gauss(x, H, A, x0, sigma):
#         return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
#     
#     def gauss_fit(x, y):
#         """
#         preform gaussian fit
# 
#         Parameters
#         ----------
#         x : list or array
#             x axis value.
#         y : list or array
#             y axis value.
# 
#         Returns
#         -------
#         popt : array
#             return H, A, x0, sigma.
#             peak height = H+A
#             peak location = x0
#             std = sigma
#             FWHM = 2.355*sigma
# 
#         """
#         mean = sum(x * y) / sum(y)
#         sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
#         popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
#         return popt
#     
#     _, freq, _ = zip(*para)
#     freq = sorted(set(freq))
#     x = [math.log(f, 2) for f in freq]
#     #x = np.arange(0,len(arr))
#     y = resp_loudness
#     popt = gauss_fit(x,y)
#     peak = popt[0]+popt[1]
#     bf = 2**popt[2]
#     band = 2.355*popt[3]
#     tone_charact = {'best frequency': bf, 'band width': band}
#     
#     return tone_charact
# =============================================================================


def tunning(resp, para, filename='', saveplot=False):
    """
    Generate tunning curve.

    Parameters
    ----------
    resp : ndarray
        Response.
    para : ndarray
        Parameters.
    filename : str, optional
        Filename
    saveplot : Boolean, optional
        Set Ture to save plot. The default is False.

    Returns
    -------
    None

    """
    
    def on_avg(arr):
        base = np.mean(arr[:500])
        arr = (arr-base)*100            
        return np.mean(arr[500:3000])
    
    def off_avg(arr):
        base = np.mean(arr[:500])
        arr = (arr-base)*100            
        return np.mean(arr[3000:5500])
    
    loud, freq, _ = zip(*para)
    loud = sorted(set(loud))
    freq = sorted(set(freq))
    resp_mesh = np.reshape(resp, (len(loud),len(freq),-1))
    resp_on = np.apply_along_axis(on_avg, 2, resp_mesh)
    resp_off = np.apply_along_axis(off_avg, 2, resp_mesh)
    
# =============================================================================
#     XX,YY = np.meshgrid(freq, loud)
#         
#     fig, ax1 = plt.subplots()
#     pcm = ax1.pcolormesh(XX, YY, resp_on, cmap='RdBu_r', norm=colors.CenteredNorm())
#     ax1.set_xscale('log')
#     ax2 = plt.subplot()
#     txt = (f'{filename}_on')
#     ax2.text(0,1.02, txt, horizontalalignment='left', transform=ax2.transAxes)
#     fig.colorbar(pcm, ax=ax1)
#     if saveplot:
#         plt.savefig(f'{filename}_on', dpi=500)
#         plt.clf()
#         plt.close()
#     else:
#         plt.show() 
#     
#     fig, ax1 = plt.subplots()
#     pcm = ax1.pcolormesh(XX, YY, resp_off, cmap='RdBu_r', norm=colors.CenteredNorm())
#     ax1.set_xscale('log')
#     ax2 = plt.subplot()
#     txt = (f'{filename}_off')
#     ax2.text(0,1.02, txt, horizontalalignment='left', transform=ax2.transAxes)
#     fig.colorbar(pcm, ax=ax1)
#     if saveplot:
#         plt.savefig(f'{filename}_off', dpi=500)
#         plt.clf()
#         plt.close()
#     else:
#         plt.show() 
# =============================================================================
    
    
    method = 'gaussian'
    xlabel = freq[::int((len(freq)-1)/10)]
    xlabel = [i/1000 for i in xlabel]
    ylabel = [int(i) for i in loud]
    Nx = len(xlabel)
    Ny = len(ylabel)
    xtick = np.arange(0.5,Nx-0.4,1)
    ytick = np.arange(0.5,Ny-0.4,1)
    
    fig, ax1 = plt.subplots()
    im = plt.imshow(resp_on, interpolation=method, origin='lower', extent=(0,Nx,0,Ny), cmap='RdBu_r', norm=colors.CenteredNorm())
    ax1.add_image(im)
    ax1.set_xticks(xtick)
    ax1.set_xticklabels(xlabel, rotation=45)
    ax1.set_yticks(ytick)
    ax1.set_yticklabels(ylabel)
    ax1.set_title(f'{filename}_onset')
    ax1.set_xlabel('Frequency kHz')
    ax1.set_ylabel('dB SPL')
    
    cax = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.03,ax1.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('mV')
    if saveplot:
        plt.savefig(f'{filename}_on.png', dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    

    method = 'gaussian'
    xlabel = freq[::int((len(freq)-1)/10)]
    xlabel = [i/1000 for i in xlabel]
    ylabel = [int(i) for i in loud]
    Nx = len(xlabel)
    Ny = len(ylabel)
    xtick = np.arange(0.5,Nx-0.4,1)
    ytick = np.arange(0.5,Ny-0.4,1)
    
    fig, ax1 = plt.subplots()
    im = plt.imshow(resp_off, interpolation=method, origin='lower', extent=(0,Nx,0,Ny), cmap='RdBu_r', norm=colors.CenteredNorm())
    ax1.add_image(im)
    ax1.set_xticks(xtick)
    ax1.set_xticklabels(xlabel, rotation=45)
    ax1.set_yticks(ytick)
    ax1.set_yticklabels(ylabel)
    ax1.set_title(f'{filename}_offset')
    ax1.set_xlabel('Frequency kHz')
    ax1.set_ylabel('dB SPL')
    
    cax = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.03,ax1.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('mV')
    if saveplot:
        plt.savefig(f'{filename}_off.png', dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    
    return int(best_freq(resp_on, para))

def baseline(resp_iter):    #correct baseline
    return (resp_iter - np.mean(resp_iter[:20*25]))*100

def psth(resp, filename, set_x_intime=False, saveplot=False):
    resp_base = np.apply_along_axis(baseline, 1, resp)
    y = np.mean(resp_base, axis=0)
    x = np.arange(0,len(y))
    err = stats.sem(resp_base, axis=0)
    
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
    [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.3) for _x in np.arange(0,5100,500)]
    [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [500,3000]]
    ax.set_title(f'{filename}_tone-PSTH')   
    ax.set_xlim(0,10000)
    ax.set_ylabel('membrane potential mV')

    if set_x_intime:
        label = np.linspace(-20,380,6)
        ax.set_xticks(np.linspace(0,10000,6),label)
        ax.set_xlabel('time ms')
    else:
        ax.set_xticks([0,500,1500,3000,5000,7000,9000])
        ax.set_xlabel('data point 2500/100ms')
        
    if saveplot:
        plt.savefig(f'{filename}_tone-PSTH.png', dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def psth_window(resp, filename, window):
    pass


def mem_V(stim, para, resp, filename='', saveplot=False):
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
    if saveplot:
        plt.savefig(f'{filename}_on', dpi=500)
        plt.clf()
    else:
        plt.show()
    
    
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
    if saveplot:
        plt.savefig(f'{filename}_off', dpi=500)
        plt.clf()
    else:
        plt.show()
    

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


def plot_avg_resp(stim, para, resp, filename='', savefig=False):
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
        if savefig:
            plt.savefig(f'{filename}_{i}.png', dpi=500)
            plt.clf()
        else:
            plt.show()
        
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
    


    
    
from TDMS import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy import stats
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import TFTool
import pandas as pd
import lsfm_analysis
from scipy.signal.windows import dpss

       
    
def pow_diff(filename, resp):
    resp_z = stats.zscore(resp)
    resp_z = signal.resample(resp_z, 1200, axis=1)
    resp_pad = np.pad(resp_z, [(0,0), (0,len(resp_z[0]))], 'constant')
    resp_fft = np.abs(np.fft.fft(resp_pad)**2)
    freq = np.fft.fftfreq(len(resp_pad[0]), d=1/600)
    mask = freq >= 0
    
    res, prop = TFTool.para_merge2(para, resp_fft, axis=2)
    target_freq = np.arange(1.0,257.0,0.5)
    oct_freq = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    label_freq = [0.0, 1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
    multi_freq = [4.0, 32.0, 256.0]
    
    idx_freq = [i for i, a in enumerate(freq) if a in oct_freq]
    idx_freq = np.array(idx_freq)
    

    #plot power difference between oct_freq and 0.5Hz
    #Currently only works when merging includs modulation rate (axis=1 or 2)
    pow_diff = []
    for pp in res:
        diff = pp[idx_freq]/pp[idx_freq-2] + pp[idx_freq]/pp[idx_freq+2]
        for i, a in enumerate(diff):
            if a >= 0:
                diff[i] = np.log(diff[i])
            elif a < 0:
                diff[i] = -1*np.log(-1*diff[i])
        pow_diff.append(diff)
    pow_diff = np.array(pow_diff)
        
    mod_rate = sorted(list(set(prop['set1'])))
    
    #get the count of each element, exclude first one (0.0)
    y_N = [[x,prop['set1'].count(x)] for x in set(prop['set1'])][1][1]
    for j in np.arange(1,7):
        x,y=[],[]
        idx = [i for i, a in enumerate(prop['set1']) if a == mod_rate[j]]
        #match element number to 9 freq in oct_freq
        for i in idx:
            y.append([prop['set2'][i]]*9)
            x.append(pow_diff[i,:])
        for k in range(y_N):
            plt.scatter(oct_freq, y[k], c=100*x[k], s=10*np.abs(x[k]), cmap='bwr')      
        ax = plt.subplot()
        txt = (f'{filename}, Modulation: {mod_rate[j]} Hz')
        ax.text(0,1.02, txt, horizontalalignment='left', transform=ax.transAxes)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Resp Freq (Hz)', fontsize=13)
        plt.ylabel(prop['parameter2'], fontsize=13)
        #plt.colorbar()
        for xc in multi_freq:
            plt.axvline(x=xc, color='r', linestyle='--', alpha=0.5)
        #plt.savefig(f'{filename}_{mod_rate[j]}.png', dpi=500)
        plt.show()
        plt.clf()
        

def pow_at_freq1(resp):
    res, prop = TFTool.para_merge(para, resp_fft, axis=0)
    power = []
    target_freq = np.arange(1.0,257.0)
    ext_freq = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    label_freq = [0.0, 1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
    oct_freq = [4.0, 32.0, 256.0]
    idx_freq = [i for i, a in enumerate(freq) if a in target_freq]

    power_at_freq=[]
    for pp in res:
        power_at_freq.append(pp[idx_freq])

    for i,a in enumerate(power_at_freq):
        plt.scatter(target_freq, a, s=12)
        plt.xscale('log')
        plt.yscale('log')
        ax = plt.subplot()
        if prop['axis'] == 1:
            txt = prop['parameter'] + '\n %.1f kHz' % prop['set'][i]
        elif prop['axis'] == 0:
            txt = prop['parameter'] + '\n %i Hz' % prop['set'][i]
        elif prop['axis'] == 2:
            txt = prop['parameter'] + '\n %.5f octave' % prop['set'][i]
        ax.text(0.95,0.85, txt, transform=ax.transAxes, horizontalalignment='right')
        for xc in label_freq:
            plt.axvline(x=xc, color='k', linestyle='--', alpha=0.3)
        for xc in oct_freq:
            plt.axvline(x=xc, color='r', linestyle='--', alpha=0.3)
        plt.show()
        
        
def pow_at_freq2(resp):
    """
    plot power at target frequency from fft_response
    
    """
    res, prop = TFTool.para_merge2(para, resp_fft, axis=2)
    target_freq = np.arange(1.0,257.0,0.5)
    oct_freq = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    label_freq = [0.0, 1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
    multi_freq = [4.0, 32.0, 256.0]
    idx_freq = [i for i,a in enumerate(freq) if a in target_freq]

    power_at_freq=[]
    for pp in res:
        power_at_freq.append(pp[idx_freq])

    for i,a in enumerate(power_at_freq):
        plt.scatter(target_freq, a, s=12)
        plt.xscale('log')
        plt.yscale('log')
        ax = plt.subplot()
        if prop['axis'] == 0:
            txt = prop['parameter'] + '\n %.1f kHz' % prop['set1'][i] + '\n %.5f octave' % prop['set2'][i]
        elif prop['axis'] == 1:
            txt = prop['parameter'] + '\n %i Hz' % prop['set1'][i] + '\n %.5f octave' % prop['set2'][i]
        else:
            txt = prop['parameter'] + '\n %i Hz' % prop['set1'][i] + '\n %.1f kHz' % prop['set2'][i]
        ax.text(0.95,0.85, txt, transform=ax.transAxes, horizontalalignment='right')
        for xc in label_freq:
            plt.axvline(x=xc, color='k', linestyle='--', alpha=0.3)
        for xc in oct_freq:
            plt.axvline(x=xc, color='r', linestyle='--', alpha=0.3)
        plt.show()
        

def plot_avg(df, resp, para):
    resp_z = stats.zscore(resp)
    resp_fft = np.abs(np.fft.fft(resp_z)**2)
    
    mean, prop = TFTool.para_merge(para, resp_z, axis=0)
    for idx, res in enumerate(mean):
        plt.plot(res)
        ax = plt.subplot()
        txt = df['date'][i]+'_'+df['#'][i] + '\n ' + \
            prop['axis']+' '+str(prop['set'][idx])
        ax.text(0.05,0.9,txt,transform=ax.transAxes, fontsize=10)
        plt.show()
        plt.clf()
    
    mean, prop = para_merge(para, resp_z, axis=1)
    for idx, res in enumerate(mean):
        plt.plot(res)
        ax = plt.subplot()
        txt = df['date'][i]+'_'+df['#'][i] + '\n ' + \
            prop['axis']+' '+str(prop['set'][idx])
        ax.text(0.05,0.9,txt,transform=ax.transAxes, fontsize=10)
        plt.show()
        plt.clf()
    
    mean, prop = para_merge(para, resp_z, axis=2)
    for idx, res in enumerate(mean):
        plt.plot(res)
        ax = plt.subplot()
        txt = df['date'][i]+'_'+df['#'][i] + '\n ' + \
            prop['axis']+' '+str(prop['set'][idx])
        ax.text(0.05,0.9,txt,transform=ax.transAxes, fontsize=10)
        plt.show()
        plt.clf()


def inv_fir(sound, fir):
    """reverse FIR filter"""
    #eliminate fft at zero (DC component)
    _fir_fft = np.delete(np.fft.fft(fir),0)
    filt = np.abs(_fir_fft)
    filt[:20] = filt[20]
    filt[-21:] = filt[-21]
    filt = np.around(filt, decimals = 12)
    r = filt[len(filt)//2]/filt
    theta = np.angle(_fir_fft, deg=False)
    inv_filt = r*np.cos(theta) + r*np.sin(theta)*1j
    inv_filt = np.fft.ifft(inv_filt)
    return np.convolve(sound, np.abs(inv_filt), 'same')


def strf(resp, cwt, filename, plot=True):
    resp_r = signal.resample(resp, 500, axis=1)
    
    f = cwt['f']
    f = f[:,0]
    wt = cwt['wt'].T[:,0]
    wt_a = []
    for w in wt:
        wt_a.append(w)
    wt_a = np.array(wt_a)
    wt_mean = wt_a.mean(axis=(0,2))
        
    """construct STRF"""
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
    
    if plot:
        plt.pcolormesh(delays_sec, f, strf, shading='nearest')
        plt.xlabel('delay time (min)', fontsize=13)
        plt.ylabel('frequency (Hz)', fontsize=13)
        plt.yscale('log')
        plt.ylim(2000,100000)
        ax = plt.subplot()
        ax.text(0.02,1.03, f'{filename}', horizontalalignment='left', transform=ax.transAxes, fontsize=13)
        plt.savefig(f'{filename}_strf.png', dpi=500)
        #plt.show()
    
    return coeff,strf


def coeff(tdms, df, loc1, loc2):
    """
    Find correlation between repeat recordings

    Parameters
    ----------
    tdms : object
        Tdms object from TDMS module
    df : Data Frame
        Pandas data frame.
    loc1 : int
        index of first recording
    loc2 : int
        index of second recording.

    Returns
    -------
    None.

    """
    t=tdms
    df_loc = loc1
    fdir = df['path'][df_loc]
    filename1 = df['date'][df_loc]+'_'+df['#'][df_loc]
    t = Tdms()
    t.loadtdms(fdir, load_sound=False)
    resp1,_ = t.get_dpk()
    
    df_loc = loc2
    fdir = df['path'][df_loc]
    filename2 = df['date'][df_loc]+'_'+df['#'][df_loc]
    t = Tdms()
    t.loadtdms(fdir, load_sound=False)
    resp2,_ = t.get_dpk()
    filename = df['date'][loc1]+'v'+df['date'][loc2]
    
    r = np.corrcoef(resp1, resp2)
    plt.imshow(r)
    plt.xlim(0,453)
    plt.ylim(0,453)
    ax = plt.subplot()
    ax.text(0.02, 1.03, f'{filename1}.vs.{filename2}', transform=ax.transAxes, fontsize=13,
            horizontalalignment='left')
    plt.savefig(f'{filename}_corr', dpi=500)
    plt.clf()
    
    R = []
    for x in np.arange(len(resp1)):
        try:
            r = np.corrcoef(resp1[x], resp2[x])
        except ValueError:
            print(x)
        else:
            R.append(r[0,1])
            

    for i,a in enumerate(R):
        plt.scatter(i, a, c='black', s=5)

    plt.ylabel('Pearson Coefficient', fontsize=12)    
    ax = plt.subplot()
    ax.text(0.02, 1.03, f'{filename1}.vs.{filename2}', transform=ax.transAxes, fontsize=13,
            horizontalalignment='left')
    plt.savefig(f'{filename}_corr2', dpi=500)
    plt.clf()
    
    
def rank(resp):
    """
    Rank response for strf

    Parameters
    ----------
    resp : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    resp_rank=[]
    for r in resp:
        pds = pd.Series(r)
        i = pds.index[(pds == pds.abs().min()) | (pds == -1*pds.abs().min())]
        pds = pds.rank()
        pds -= pds[i[0]]
        pds = pds.to_numpy()
        resp_rank.append(pds)
        
    return resp_rank
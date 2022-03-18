from TDMS_ver1 import Tdms
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
from scipy.signal.windows import dpss

       
    
def pow_diff(filename, resp, para):
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
        diff = diff/2
# =============================================================================
#         for i, a in enumerate(diff):
#             if a >= 0:
#                 diff[i] = np.log(diff[i])
#             elif a < 0:
#                 diff[i] = -1*np.log(-1*diff[i])
# =============================================================================
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
        plt.savefig(f'{filename}_{mod_rate[j]}.png', dpi=500)
        #plt.show()
        plt.clf()
        

def pow_at_freq1(resp, para, filename):
    resp_z = stats.zscore(resp)
    resp_z = signal.resample(resp_z, 1200, axis=1)
    resp_pad = np.pad(resp_z, [(0,0), (0,len(resp_z[0]))], 'constant')
    resp_fft = np.abs(np.fft.fft(resp_pad)**2)
    freq = np.fft.fftfreq(len(resp_pad[0]), d=1/600)
    mask = freq >= 0
    
    res, prop = TFTool.para_merge(para, resp_fft, axis=0)
    power = []
    target_freq = np.arange(1.0,257.0)
    oct_freq = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    label_freq = [1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
    multi_freq = [4.0, 32.0, 256.0]
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
            plt.axvline(x=xc, color='k', linestyle='--', alpha=0.5)
        for xc in multi_freq:
            plt.axvline(x=xc, color='r', linestyle='--', alpha=0.5)
        plt.show()
        
        
def pow_at_freq2(resp, para, filename):
    """
    plot power at target frequency from fft_response
    
    """
    resp_z = stats.zscore(resp)
    resp_z = signal.resample(resp_z, 1200, axis=1)
    resp_pad = np.pad(resp_z, [(0,0), (0,len(resp_z[0]))], 'constant')
    resp_fft = np.abs(np.fft.fft(resp_pad)**2)
    freq = np.fft.fftfreq(len(resp_pad[0]), d=1/600)
    mask = freq >= 0
    
    res, prop = TFTool.para_merge2(para, resp_fft, axis=2)
    target_freq = np.arange(1.0,257.0,0.5)
    oct_freq = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    label_freq = [1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
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
            txt = prop['parameter1'] + '\n %.1f kHz' % prop['set1'][i] \
                + '\n'+prop['parameter2']+'\n %.5f octave' % prop['set2'][i]
        elif prop['axis'] == 1:
            txt = prop['parameter1'] + '\n %i Hz' % prop['set1'][i] \
                + '\n'+prop['parameter2'] + '\n %.5f octave' % prop['set2'][i]
        else:
            txt = prop['parameter1'] + '\n %i Hz' % prop['set1'][i] \
                + '\n'+prop['parameter2'] +'\n %.1f kHz' % prop['set2'][i]
        ax.text(0.95,0.95, txt, transform=ax.transAxes, \
                va='top', ha='right')
        for xc in label_freq:
            plt.axvline(x=xc, color='k', linestyle='--', alpha=0.5)
        for xc in multi_freq:
            plt.axvline(x=xc, color='r', linestyle='--', alpha=0.5)
        plt.savefig(f'{filename}_{i}.png', dpi=500)
        #plt.show()
        plt.clf()
        

def plot_avg(df, resp, para):
    resp_z = stats.zscore(resp)
    #resp_fft = np.abs(np.fft.fft(resp_z)**2)
    
    mean, prop = TFTool.para_merge(para, resp_z, axis=0)
    for idx, res in enumerate(mean):
        plt.plot(res)
        ax = plt.subplot()
        txt = df['date'][idx]+'_'+df['#'][idx] + '\n ' + \
            prop['axis']+' '+str(prop['set'][idx])
        ax.text(0.05,0.9,txt,transform=ax.transAxes, fontsize=10)
        plt.show()
        plt.clf()
    
    mean, prop = TFTool.para_merge(para, resp_z, axis=1)
    for idx, res in enumerate(mean):
        plt.plot(res)
        ax = plt.subplot()
        txt = df['date'][idx]+'_'+df['#'][idx] + '\n ' + \
            prop['axis']+' '+str(prop['set'][idx])
        ax.text(0.05,0.9,txt,transform=ax.transAxes, fontsize=10)
        plt.show()
        plt.clf()
    
    mean, prop = TFTool.para_merge(para, resp_z, axis=2)
    for idx, res in enumerate(mean):
        plt.plot(res)
        ax = plt.subplot()
        txt = df['date'][idx]+'_'+df['#'][idx] + '\n ' + \
            prop['axis']+' '+str(prop['set'][idx])
        ax.text(0.05,0.9,txt,transform=ax.transAxes, fontsize=10)
        plt.show()
        plt.clf()


def inv_fir(sound, fir):
    """reverse FIR filter"""
    #eliminate fft at zero (DC component)
    #_fir_fft = np.delete(np.fft.fft(fir),0)
    _fir_fft = np.fft.fft(fir)
    theta = np.angle(_fir_fft, deg=False)
    dc = _fir_fft[0]
    filt = np.abs(_fir_fft)
    filt[:20] = filt[20]
    filt[-19:] = filt[-20]
    filt = np.around(filt, decimals = 12)
    r = filt[len(filt)//2]/filt
    #theta = np.angle(_fir_fft, deg=False)
    inv_filt = r*np.cos(theta) + r*np.sin(theta)*1j
    inv_filt = np.real(np.fft.ifft(inv_filt))
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


def coeff(df, loc1, loc2):
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
    
    avg = np.around(np.mean(R), decimals=4) 
        
    plt.ylabel('Pearson Coefficient', fontsize=12)    
    ax1 = plt.subplot()
    ax1.text(0.02, 1.03, f'{filename1}.vs.{filename2}', transform=ax1.transAxes, fontsize=13,
            horizontalalignment='left')
    ax2 = plt.subplot()
    ax2.text(0.95, 0.95, f'average: {avg}', transform=ax2.transAxes, fontsize=12,
            horizontalalignment='right')
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


class RespAtFreq():
    def __init__(self):
        self.para_s, self.wt_s, self.resp_s, self.stim_s = [],[],[],[]
    
    def mod_reduction(self, stim, resp, para, df, df_loc, cwt):            
        _f = cwt['f']
        self.f = _f[:,0]
        wt = cwt['wt'].T[:,0]
        
        _, _, mod, _ = zip(*para)
        #use mod_rate at 1.0, 2.0, 8.0, 16.0 to avoid response contamination
        slow = [i for i, a in enumerate(mod) if a >=1.0 and a <= 16.0]
        for i in slow:
            self.para_s.append(para[i])
            self.wt_s.append(wt[i])
            self.resp_s.append(resp[i])
            self.stim_s.append(stim[i])
        
        self.stim_s = np.array(self.stim_s)
        self.resp_s = np.array(self.resp_s)
        self.dir = df['path'][df_loc]
        self.filename = df['date'][df_loc]+'_'+df['#'][df_loc]


    def resp_at_freq(self, nth_freq=False, find_slope=True, plot=False):
        def nth_cross(peaks):    
            """
            set the n value in crossing to show first n-th response 
            when stimulus crosses target frequency
            """
            crossing = [0,1,2,3,4,5]
            while len(crossing) > len(peaks):
                crossing.pop()
            return crossing
            
        
        def find_slopes(idx, x):
            fs = 200000
            """cwt decimation rate is 800 to 250Hz"""
            x = x*8
            b,a = signal.butter(2, 6000, btype='low', fs=fs)
            h = signal.hilbert(self.stim_s[idx])
            phase = np.unwrap(np.angle(h))
            inst_freq = np.diff(phase) / (2*np.pi) * fs
            inst_freq = signal.filtfilt(b,a,inst_freq)
            slope = np.diff(inst_freq)
            
            return slope[x]
                
        """
        set frequencies of interest
        """
        _target_freq = [3,6,12,24,36,48,60,72,96]
        self.target_freq = [i*1000 for i in _target_freq]
        
        self.windows, self.slopes, self.nth = [],[],[]
        
        for freq in self.target_freq:
            i_freq = TFTool.find_nearest(freq, self.f)       
            peak_store = []
            #peak_pre, peak_post = [],[]
            windows = []
            slopes = []
            """depend on the length of crossing"""
            windows_nth = [[] for n in range(6)]

            
            for idx,spectrum in enumerate(self.wt_s):
                peaks,peak_paras = signal.find_peaks(spectrum[i_freq], prominence=0.2)
                #p1,_ = signal.find_peaks(spectrum[i_freq-1], prominence=0.2)
                #p2,_ = signal.find_peaks(spectrum[i_freq+1], prominence=0.2)
                peak_store.append(peaks*100)
                #peak_pre.append(p1*100)
                #peak_post.append(p2*100)
                

                if len(peaks) != 0:
                    crossing = nth_cross(peaks)
                    
                    for i,x in enumerate(peaks):                            
                        x = x*100
                        window = self.resp_s[idx][x-1250:x+3750]
                        windows.append(window)
                                                    
                        if find_slope:
                            slopes.append(find_slopes(idx, x))
                        
                        if nth_freq:
                            for n in crossing:
                                if n == i:
                                    windows_nth[n].append(window)


            
            self.windows.append(windows)
            self.slopes.append(slopes)
            self.nth.append(windows_nth)
            
            if plot:
                plt.plot(np.mean(windows, axis=0))
                plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                ax = plt.subplot()
                txt = (f'{self.filename}-{freq} Hz. Averaged from {len(windows)}')
                ax.text(0,1.02, txt, horizontalalignment='left', transform=ax.transAxes)
                plt.savefig(f'resp_at_freq-{self.filename}-{freq}.png', dpi=500)
                plt.clf()
            
            if nth_freq and plot:
                colors = plt.cm.OrRd(np.linspace(1,0.3,len(windows_nth)))
                for n in range(len(windows_nth)):
                    nth_mean = np.mean(windows_nth[n], axis=0)
                    nth_mean = nth_mean - nth_mean[1250]
                    plt.plot(nth_mean, c=colors[n], label='%s th' %str(n+1))
                    
                plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)  
                ax = plt.subplot()
                txt = (f'{self.filename}-{freq} Hz.')
                ax.text(0,1.02, txt, horizontalalignment='left', transform=ax.transAxes)    
                plt.savefig(f'nth_cross-{self.filename}-{freq}.png', dpi=500)
                plt.clf()

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
        index of first recording in df
    loc2 : int
        index of second recording in df

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
    resp : ndarray
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
            b,a = signal.butter(3, 150, btype='low', fs=fs)
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
        
        self.windows, self.slopes, self.nth, self.latencies, self.averages = [],[],[],[],[]
        
        for freq in self.target_freq:
            i_freq = TFTool.find_nearest(freq, self.f)       
            peak_store = []
            #peak_pre, peak_post = [],[]
            windows = []
            slopes = []
            latencies = []
            averages = []
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
                        avg = window - window[1250]
                        peak = max(avg) + min(avg)
                        averages.append(peak)
                        latencies.append(x)
                                                    
                        if find_slope:
                            slopes.append(find_slopes(idx, x))
                        
                        if nth_freq:
                            for n in crossing:
                                if n == i:
                                    windows_nth[n].append(window)


            
            self.windows.append(windows)
            self.slopes.append(slopes)
            self.nth.append(windows_nth)
            self.latencies.append(latencies)
            self.averages.append(averages)
            
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


class Psth():
    def __init__(self, resp, para, filename):
        #exclude carrier less than 3kHz and puretone
        p_t, r_t = [],[]
        for i,p in enumerate(para):
            if p[0] < 3.0:
                pass
            elif p[2] == 0.0:
                pass
            else:
                p_t.append(p)
                r_t.append(resp[i])
        
        self.resp = r_t
        self.para = p_t
        self.filename = filename
        _para = np.swapaxes(np.array(self.para),0,1)
        self.mod_label = sorted(set(_para[2][:]))
        self.cf_label = sorted(set(_para[0][:]))
        self.bw_label = sorted(set(_para[1][:]))
        self.features = pd.DataFrame()
        
    """reutrn *100 to switch from LabView volt to real mV scale"""
    def baseline(resp_iter):    #correct baseline
        return (resp_iter - np.mean(resp_iter[:50*25]))*100
    
    def baseline_zero(resp_iter):   #fix sound onset to zero
        return (resp_iter - resp_iter[50*25])*100
    
    def psth_all(self, set_x_intime=False, saveplot=False):
        """
        Generates PSTH using all lsfm response.

        Parameters
        ----------
        set_x_intime : boolean
            Ture to set x-axis in second instead of data point. The default is False.
        saveplot : boolean
            Saveplot. The default is False.

        Returns
        -------
        None.

        """
        
        
        resp = np.array(self.resp)
        resp_base = np.apply_along_axis(Psth.baseline, 1, resp)
        
        y = np.mean(resp_base, axis=0)
        x = np.arange(0,len(y))
        err = stats.sem(resp_base, axis=0)
        
        plt.plot(x,y)
        plt.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
        plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=38750, color='k', linestyle='--', alpha=0.5)
        label = list(np.round(np.linspace(0, 2.0, 21), 2))
        if set_x_intime:
            plt.xticks(np.linspace(0,50000,21),label)
        else:
            plt.xticks(np.linspace(0,50000,21))
            plt.xticks(rotation = 45)
        ax = plt.subplot()
        txt = (f'{self.filename}-PSTH')
        ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
        if saveplot:
            plt.savefig(f'{self.filename}-PSTH.png', dpi=500)
            plt.clf()
        else:
            plt.show()
    
    def psth_para(self, plot=False, saveplot=False) -> dict:
        """
        Ploting PSTH seperate with each parameters.

        Parameters
        ----------
        plot : boolean
            Show plots. The default is False.
        saveplot : boolean
            Save plots. The default is False.

        Returns
        -------
        dict
            Return responses seperated by parameters: 
            'modrate', 'centerfreq', 'bandwidth'.

        """
        
        
        resp = np.array(self.resp)
        _para = np.swapaxes(np.array(self.para),0,1)
        para_mod = _para[2][:]
        para_cf = _para[0][:]
        para_band = _para[1][:]
        
        resp_mod, resp_cf, resp_band=[],[],[]
        
        for mod in self.mod_label:
            temp = []
            for i, p in enumerate(para_mod):
                if p == mod:
                    temp.append(Psth.baseline(resp[i]))     #resp with same mod_rate
            resp_mod.append(temp)       #resp seperated by mod_rate
    
        for cf in  self.cf_label:
            temp = []
            for i, p in enumerate(para_cf):
                if p == cf:
                    temp.append(Psth.baseline(resp[i]))
            resp_cf.append(temp)
    
        for band in self.bw_label:
            temp = []
            for i, p in enumerate(para_band):
                if p == band:
                    temp.append(Psth.baseline(resp[i]))
            resp_band.append(temp)
            
        if plot:
            for i in range(len(self.mod_label)):
                y = np.mean(resp_mod[i], axis=0)
                x = np.arange(0,len(y))
                err = stats.sem(resp_mod[i], axis=0)
                
                plt.plot(x,y)
                plt.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                plt.axvline(x=38750, color='k', linestyle='--', alpha=0.5)
                label = list(np.round(np.linspace(0, 2.0, 11), 2))
                plt.xticks(np.linspace(0,50000,11),label)
                ax = plt.subplot()
                txt = (f'{self.filename}-mod {self.mod_label[i]} Hz')
                ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
                
                if saveplot:
                    plt.savefig(f'{self.filename}-mod {self.mod_label[i]} Hz.png', dpi=500)
                    plt.clf()
                else:
                    plt.show()
            
            for i in range(len(self.cf_label)):
                y = np.mean(resp_cf[i], axis=0)
                x = np.arange(0,len(y))
                err = stats.sem(resp_cf[i], axis=0)
                
                plt.plot(x,y)
                plt.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                plt.axvline(x=38750, color='k', linestyle='--', alpha=0.5)
                label = list(np.round(np.linspace(0, 2.0, 11), 2))
                plt.xticks(np.linspace(0,50000,11),label)
                ax = plt.subplot()
                txt = (f'{self.filename}-cf {self.cf_label[i]} Hz')
                ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
                
                if saveplot:
                    plt.savefig(f'{self.filename}-cf {self.cf_label[i]} kHz.png', dpi=500)
                    plt.clf()
                else:
                    plt.show()
            
            for i in range(len(self.bw_label)):
                y = np.mean(resp_band[i], axis=0)
                x = np.arange(0,len(y))
                err = stats.sem(resp_band[i], axis=0)
                
                plt.plot(x,y)
                plt.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
                plt.axvline(x=1250, color='k', linestyle='--', alpha=0.5)
                plt.axvline(x=38750, color='k', linestyle='--', alpha=0.5)
                label = list(np.round(np.linspace(0, 2.0, 11), 2))
                plt.xticks(np.linspace(0,50000,11),label)
                ax = plt.subplot()
                txt = (f'{self.filename}-bdwidth {self.bw_label[i]} kHz')
                ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
                
                if saveplot:
                    plt.savefig(f'{self.filename}-bdwidth {self.bw_label[i]} kHz.png', dpi=500)
                    plt.clf()
                else:
                    plt.show()
            
        #return resp grouped by parameters
        return {'modrate':resp_mod, 'centerfreq':resp_cf,
                'bandwidth':resp_band}
    
    
    def psth_correlation(self, saveplot=False):
        psth_a = Psth.psth_all(self)
        psth_p = Psth.psth_para(self)
        
        '''coeff'''
        mod_coeff, cf_coeff, bw_coeff = [],[],[]
     
        mod = psth_p['modrate']
        for i in range(len(mod)):
            psth_mod = np.mean(mod[i], axis=0)    
            mod_coeff.append(stats.pearsonr(psth_a, psth_mod)[0])
        plt.plot(mod_coeff)
        plt.xticks(list(range(len(mod))), self.mod_label)
        ax = plt.subplot()
        txt = (f'{self.filename}_Modrate_coeff')
        ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
        if saveplot:
            plt.savefig(f'{self.filename}_PSTH_ModRate_Coeff.png', dpi=500)
            plt.clf()
        else:
            plt.show()
        
        cf = psth_p['centerfreq']
        for i in range(len(cf)):
            psth_cf = np.mean(cf[i], axis=0)
            cf_coeff.append(stats.pearsonr(psth_a, psth_cf)[0])
        plt.plot(cf_coeff)
        plt.xticks(list(range(len(cf))), self.cf_label)
        plt.xticks(rotation = 45)
        ax = plt.subplot()
        txt = (f'{self.filename}_centerfreq_coeff')
        ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
        if saveplot:
            plt.savefig(f'{self.filename}_PSTH_CenterFreq_Coeff.png', dpi=500)
            plt.clf()
        else:
            plt.show()
        
        bw = psth_p['bandwidth']
        for i in range(len(bw)):
            psth_bw = np.mean(bw[i], axis=0)
            bw_coeff.append(stats.pearsonr(psth_a, psth_bw)[0])
            plt.show()
        plt.plot(bw_coeff)
        plt.xticks(list(range(len(bw))), self.bw_label)
        plt.xticks(rotation = 45)
        ax = plt.subplot()
        txt = (f'{self.filename}_bandwidth_coeff')
        ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
        if saveplot:
            plt.savefig(f'{self.filename}_PSTH_BandWidth_Coeff.png', dpi=500)
            plt.clf()
        else:
            plt.show()
            
    def psth_trend(self, saveplot=False, tuning=None, **kwargs) -> None:
        """
        Generate average potential vs base. Traces seperated by group.

        Parameters
        ----------
        saveplot : boolean
            Set ture to save plot. The default is False.
        tuning : tuple, optional
            Use tuning=(start,end) to specify the frequency range in kHz to include
            when averaging carrier frequency.
        **kwargs : str
            window = (start,end) to specify the resposne range in datapoint
            locaiton = 'onset', 'second, 'plateau', 'offset'
            onset: 0-0.4sec, second: 0.4-0.8sec, plateau: 0.8-1.6sec, offset: 1.6-2.0sec
           
        
        Returns
        -------
        None.

        """
        
        _para = np.swapaxes(np.array(self.para),0,1)
        para_cf = _para[0][:]
        para_band = _para[1][:]
        para_mod = _para[2][:]
        para_name = ['Center Freq', 'Band Width', 'Mod Rate']
        label_list = [self.cf_label, self.bw_label, self.mod_label]
        para_list = [para_cf, para_band, para_mod]
        
        from itertools import permutations
        aranges = []
        
        #generate combinations
        for i in permutations(range(3),3):
            aranges.append(i)
        
        #e.g. arange=[0,2,1]: grouped by cf, plot mV versus mod rate(base),
        #average over bandwidth
        for arange in aranges:
            group = para_name[arange[0]]
            base = para_name[arange[1]]
            volt = para_name[arange[2]]
            
            samegroup=[]    #reset in different arangement
            for g in label_list[arange[0]]:               
                samebase=[]     #reset in each group
                for b in label_list[arange[1]]:
                    resp_incategory=[]      #reset in every base
                    for i,p in enumerate(self.para):                                                                 
                        if p[arange[0]] == g and p[arange[1]] == b:

                            _resp = Psth.baseline(self.resp[i])
                            
                            set_window = kwargs.get('window')
                            if set_window:
                                if(set_window[0]>set_window[1]):
                                    set_window[0], set_window[1] = set_window[1], set_window[0]
                                if(min(set_window) < 0):
                                    raise ValueError('Cannot start before zero')
                                if(max(set_window) > len(_resp)):
                                    raise ValueError('Exceed data range')
                            
                                _resp = _resp[set_window[0]:set_window[1]]
                                
                            set_location = kwargs.get('location')
                            if(set_location == 'onset'):
                                _resp = _resp[0:10000]
                            elif(set_location == 'second'):
                                _resp = _resp[10000:20000]
                            elif(set_location == 'plateau'):
                                _resp = _resp[20000:40000]
                            elif(set_location == 'offset'):
                                _resp = _resp[40000:]
                           
                            #exclude resp if not in tuning range
                            if arange[2] == 0 and tuning != None:
                                if p[arange[2]] < float(tuning[0]) or p[arange[2]] > float(tuning[1]):
                                    pass
                                else:
                                    resp_incategory.append(_resp)
                            else:
                                resp_incategory.append(_resp)

                    if resp_incategory:
                        v_mean = np.mean(resp_incategory, axis=1)
                        samebase.append([g,b,np.mean(v_mean),np.std(v_mean)])
                samegroup.append(samebase)
                
            colors = plt.cm.rainbow(np.linspace(0.3,1,len(samegroup)))
            
            for i,gp in enumerate(samegroup):              
                x,y,err=[],[],[]
                for ii in gp:
                    x.append(ii[1])
                    y.append(ii[2])
                    err.append(ii[3])
                try:
                    plt.errorbar(x,y,yerr=err, color=colors[i], capsize=(4), marker='o', label=f'{group}-{gp[0][0]}')
                except IndexError:
                    pass
                plt.xscale('symlog')                
                plt.xlabel(f'{base}')
                plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
            if saveplot:
                plt.savefig(f'{self.filename}_function_{group}-{base}.png', \
                            dpi=500, bbox_inches='tight')
                plt.clf()
            else:
                plt.show()
        

    def psth_window(self, window, featname, saveplot=False, savenotes=False):
        """
        Generate PSTH for every parameters within the range of interest.

        Parameters
        ----------
        window : Tuple
            Use (start,end) to specify the range in data point.
        featname : str
            The name of the feature of interest.
        saveplot : TYPE
            Save plot. The default is False.
        savenotes : TYPE
            Save Pandas dataframe to csv file. The default is False.

        Returns
        -------
        None.

        """
        
        try: self.features
        except NameError:
            self.features = pd.DataFrame()
        
        #be aware the resp in resp_list is returned from psth_para which went through
        #baseline correction thus already scaled to real value. 
        para_dict = Psth.psth_para(self)
        resp_list = [para_dict['centerfreq'],para_dict['bandwidth'],para_dict['modrate']]
        label_list = [self.cf_label, self.bw_label, self.mod_label]
        para_name = ['center_freq (KHz)', 'bandwidth (octave)', 'mod_rate (Hz)']
        
        for par in range(3): #parameter
            x,y = [],[]    
            for con in range(len(label_list[par])): #condition
                def slicing(arr, start, end):
                    return arr[start:end]
                resp_window = np.apply_along_axis(slicing, 1, \
                        resp_list[par][con], start=window[0], end=window[1])  
                resp_incondition = np.mean(resp_window, axis=1)
                _x = label_list[par][con]
                _y = np.mean(resp_incondition, axis=0)
                x.append(_x)
                y.append(_y)
                temp = pd.DataFrame({'x':[_x], 'y':[_y], 'base':[f'{para_name[par]}'], \
                    'feature':[f'{featname}'], 'start':[window[0]], 'end':[window[1]]})
                self.features = pd.concat((self.features, temp), axis=0)            
            
            plt.plot(x,y, marker='o')
            plt.xlabel(f'{para_name[par]}')
            plt.ylabel('mV')
            if par == 0 or par == 2:
                plt.xscale('log')
            ax = plt.subplot()
            txt = (f'{self.filename}_{featname}, range:{window[0]} to {window[1]}')
            ax.text(0,1.03, txt, horizontalalignment='left', transform=ax.transAxes)
            
            if saveplot:
                plt.savefig(f'{self.filename}_{para_name[par]}_Feature_{featname}.png', dpi=500)
                plt.clf()
            else:
                plt.show()
        if savenotes:
            self.features.to_csv(f'{self.filename}--feature_notes.csv', index=False)

def fake_sound(stim, para):
    for p in para:
        fc = para[0]
        
        
        
        
        
        
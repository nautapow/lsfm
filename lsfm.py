from TDMS_ver5 import Tdms_V1, Tdms_V2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from scipy import signal
from scipy import stats
import scipy.io
import TFTool
import pandas as pd
from scipy.signal.windows import dpss
import math
import lsfm_slope


def get_info(list_of_interests):
    if os.path.isfile('lsfm_cell_note.xlsx'):
        df_info = pd.read_excel('lsfm_cell_note.xlsx')
        for i in list_of_interests:
            if i not in list(df_info['index']):
                print(f'index {i} not in record\n')
        
        print('lsfm_cell_note.xlsx will be renamed to update new entry')
        try:
            os.rename('lsfm_cell_note.xlsx', 'lsfm_cell_note_old.xlsx')
        except FileExistsError:
            os.remove('lsfm_cell_note_old.xlsx')
            os.rename('lsfm_cell_note.xlsx', 'lsfm_cell_note_old.xlsx')
               
    df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
    
    filename, index, version, mouseID, site, tuning, index_tone, bf, bandwidth, band_left, band_right=[],[],[],[],[],[],[],[],[],[],[]
    for idx in list_of_interests:
        filename.append(df['filename'][idx])
        index.append(idx)
        version.append(df['Py_version'][idx])
        mouse = df['mouse_id'][idx]
        mouseID.append(mouse)
        site.append(df['patch_site'][idx])
        tuning_tone = df['tuning_from'][idx]
        tuning.append(tuning_tone)
        index_tone.append(df[df['filename']==tuning_tone].index.values[0])
        
        data = np.load(f'{tuning_tone}.npy', allow_pickle=True)
        resp = data.item()['resp']
        para = data.item()['para']
   
        from tone import PureTone  
        tone1 = PureTone(resp, para, mouse, tuning_tone)
        tone1.get_bf()
        tone1.get_bandwidth()
        bf.append(tone1.bf)
        bandwidth.append(tone1.bandwidth)
        band_left.append(tone1.band_left)
        band_right.append(tone1.band_right)
        
    data = pd.DataFrame({'filename':filename, 'index':index, 'version':version, 'mouseID':mouseID, 'patch_site':site,
            'tuning_tone':tuning, 'index_tone':index_tone, 'best_frequency':bf, 'bandwidth':bandwidth, 'band_left':band_left, 'band_right':band_right})
    data.to_excel('lsfm_cell_note.xlsx')
    
        


def baseline(arr):
    arr = np.array(arr)
    
    return (arr - np.mean(arr[:50*25]))*100
       
    
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
    
# =============================================================================
#     f = cwt['f']
#     f = f[:,0]
#     wt = cwt['wt'].T[:,0]
# =============================================================================
    
    f = cwt['f']
    wt = cwt['wt']
    f = f[::-1]
    wt = [i[::-1] for i in wt]
    
    wt_a = []
    for w in wt:
        wt_a.append(w)
    wt_a = np.array(wt_a)
    wt_mean = wt_a.mean(axis=(0,2))
        
    """construct STRF"""
    t_for,t_lag = 0.1,0.4
    fs = 25000
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
    t = Tdms_V2()
    t.loadtdms(fdir, load_sound=False)
    resp1,_ = t.get_dpk()
    
    df_loc = loc2
    fdir = df['path'][df_loc]
    filename2 = df['date'][df_loc]+'_'+df['#'][df_loc]
    t = Tdms_V2()
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


def resp_freq(stim, resp, para, lags, best_freq):
    """
    Get information of stim and lagged responses when stim cross best_freq

    Parameters
    ----------
    stim : TYPE
        DESCRIPTION.
    resp : TYPE
        DESCRIPTION.
    para : TYPE
        DESCRIPTION.
    lags : TYPE
        DESCRIPTION.
    best_freq : float
        DESCRIPTION.

    Returns
    -------
    resp_at_freq : list of dictionary
        '#':# of stim, 'para': stim para, 'location': location of crossing in data point, 
        'resp_lag': resps at lags seperated by stim#, 'slope':stim slope at crossing seperated by stim#.

    """
    
    idx=[i for i, a in enumerate(para) if a[2] not in [0.0,16.0,64.0,128.0]]
    fs=25000
    lag_points = [int(n*(fs/1000)) for n in lags]
    resp_at_freq = []
    
    for i in idx:
        """iterate through all stimuli and responses"""
        inst_freq, slopes = lsfm_slope.get_stimslope(stim[i])
        cross = np.abs(np.diff(np.sign(inst_freq - best_freq)))
        x_idx = [i for i,a in enumerate(cross) if a > 0]
        
        resp_base_correct = lsfm_slope.baseline(resp[i])
        
        if x_idx:
            resp_lag_each_stim=[]
            slope_each_stim=[]
            for x in x_idx:
                """iterate x when stim cross best freq"""
                resp_lag_each_stim.append([resp_base_correct[x+lag] for lag in lag_points])
                slope_each_stim.append(slopes[x])
                
            case = {'#':i, 'para':para[i][:3], 'location':x_idx, 'resp_lag':resp_lag_each_stim, 'slope':slope_each_stim}
            resp_at_freq.append(case)
        
    return resp_at_freq
        

def resp_freq_restrain(stim, resp, para, lags, bf):
    """ get resp when cf is around bf with fixed mod_rate"""
    cf,_,_,_ = zip(*para)
    cf = np.array(sorted(set(cf)))
    tgt_freq_idx = np.argmin(np.abs(cf*1000-bf))
    
    idx = [i for i,a in enumerate(para) if a[0]==cf[tgt_freq_idx] and a[2] == 8.0]
    fs=25000
    lag_points = [int(n*(fs/1000)) for n in lags]
    resp_at_freq_restrain = []
    
    for i in idx:
        """iterate through all stimuli and responses"""
        inst_freq, slopes = lsfm_slope.get_stimslope(stim[i])
        cross = np.abs(np.diff(np.sign(inst_freq - bf)))
        x_idx = [i for i,a in enumerate(cross) if a > 0]
        
        resp_base_correct = lsfm_slope.baseline(resp[i])
        
        if x_idx:
            resp_lag_each_stim=[]
            slope_each_stim=[]
            for x in x_idx:
                """iterate through stim when cross best freq at each x"""
                resp_lag_each_stim.append([resp_base_correct[x+lag] for lag in lag_points])
                slope_each_stim.append(slopes[x])
                
            case = {'#':i, 'para':para[i][:3], 'location':x_idx, 'resp_lag':resp_lag_each_stim, 'slope':slope_each_stim}
            resp_at_freq_restrain.append(case)
        
    return resp_at_freq_restrain


def nXing_cell(resp_at_freq_cell):
    """
    plot n_xing of all cells.
    use resp_at_freq_cell to listing a dictionary contain resp_at_freq from reso_freq_restrain and best_lag from 

    Parameters
    ----------
    resp_at_freq_cell : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    resp_best_lag = []
    
    for cell in resp_at_freq_cell:
        best_lag = cell['best_lag']
        resp_at_freq = cell['resp_at_freq']
        
        for n_stim in resp_at_freq:
            #swap axis to make (lags, N_cross)
            resp_at_lag = np.swapaxes(n_stim['resp_lag'], 1, 0)[best_lag]
            resp_best_lag.append(resp_at_lag)
    
    return resp_best_lag
        


def at_freq_lag(resp_at_freq, filename='', plot=True, saveplot=False):
    """
    plot instant response at each lag time after stimulus cross best frequency
    (for bf_Xing)

    Parameters
    ----------
    resp_at_freq : list of dictionary
        list generated from resp_freq().

    Returns
    -------
    a_mean : ndarray
        averaged resp at lags
    best_lag : int
        best lag according to cross bf
    """
    
    n_stim = len(resp_at_freq)
    
    all_resp_lag=[]
    for n in range(n_stim):
        slope = resp_at_freq[n]['slope']
        resp_lag = resp_at_freq[n]['resp_lag']
        
        for i in range(len(slope)):
            """iter through all crossing within a stimulus"""
            """restrain i if only want to get certain cross e.g. first crossing i==0"""
            
            all_resp_lag.append(resp_lag[i])
            
    a_mean = np.mean(all_resp_lag, axis=0)
    a_std = stats.sem(all_resp_lag, axis=0)
    local_max = signal.argrelextrema(a_mean, np.greater, order=20)
    try:
        best_lag = local_max[0][0]
    except:
        best_lag = np.nan
        
    x = range(len(a_mean))
    fig, ax = plt.subplots()
    ax.plot(x, a_mean)
    ax.fill_between(x, a_mean+a_std, a_mean-a_std, color='orange', alpha=0.5)
    #ax.set_title(f'{filename}_allX_outside BF')
    ax.set_title(f'{filename}_allX_best_lag:{best_lag*2}ms', fontsize=12)
    ax.set_xlim(0,50)
    ax.set_xticks([0,10,20,30,40,50])
    ax.set_xticklabels([0,20,40,60,80,100])
    ax.set_xlabel('lag (ms)', fontsize=16)
    ax.set_ylabel('potential at lag (mV)', fontsize=12)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='x', which='major', labelsize=12)
    
    if saveplot:
        plt.savefig(f'{filename}_bf-lag_allX.png', dpi=500, bbox_inches='tight')
        plt.savefig(f'{filename}_bf-lag_allX.pdf', dpi=500, format='pdf', bbox_inches='tight')
        #plt.savefig(f'{filename}_outside-BF_allX.png', dpi=500, bbox_inches='tight')
        #plt.savefig(f'{filename}_outside-BF_allX.pdf', dpi=500, format='pdf', bbox_inches='tight')
        if plot:
            plt.show()
        plt.clf()
        plt.close(fig)
    if plot:
        plt.show()
        plt.close(fig)
    
    return a_mean, a_std, best_lag
    
def at_freq_ncross(resp_at_freq, best_lag):
    n_stim = len(resp_at_freq)

    ncross_stim=[]
    for n in range(n_stim):
        slope = resp_at_freq[n]['slope']
        resp_lag = resp_at_freq[n]['resp_lag']
        
        ncross_each_stim=[]
        for i in range(len(slope)):
            ncross_each_stim.append(resp_lag[i][best_lag])
        
        ncross_stim.append(ncross_each_stim)
        
    ncross_comb=[]
    for n in ncross_stim:
        ncross_comb = TFTool.list_comb(n, ncross_comb)
    ncross_comb = list(ncross_comb)
    
    ncross_avg = np.nanmean(ncross_comb, axis=1)
    '''std currently not working with zip_longest'''
    ncross_std = np.nanstd(ncross_comb, axis=1)    
    
    return ncross_stim, ncross_avg
       
# =============================================================================
# def fsc_modrate(stim, resp, para, lag, filename=None, saveplot=False):
#     modrate = [i[2] for i in para]
#     mr_set = sorted(set(modrate))
#     mr_set.remove(0.0)
#     for mr in mr_set:
#         data=[[],[],[]]
#         for i,p in enumerate(modrate):
#             if p == mr:
#                 data = np.concatenate((data,single_freq_slope(stim[i], resp[i], lag)), axis=1)
#         
#         x_edges = [3000,4240,6000,8480,12000,16970,24000,33940,48000,67880,96000] 
#         y_edges = np.linspace(-20,20,51)    
#             
#         ret = stats.binned_statistic_2d(data[0], data[1], data[2], 'mean', bins=[x_edges,y_edges])
#         XX, YY = np.meshgrid(x_edges,y_edges)
#         #XX, YY = np.meshgrid(ret[1], ret[2])
#         fig, ax1 = plt.subplots()
#         #divnorm=colors.TwoSLopeNorm(vmin=-10., vcenter=0., vmax=10.)
#         pcm = ax1.pcolormesh(XX, YY, ret[0].T, cmap='RdBu_r', norm=colors.CenteredNorm())
#         ax1.set_xscale('log')
#         
#         ax2 = plt.subplot()
#         txt = (f'{filename}-modrate:{mr}-Lag:{lag}ms')
#         ax2.text(0,1.02, txt, horizontalalignment='left', transform=ax2.transAxes)
#         fig.colorbar(pcm, ax=ax1)
#         if saveplot:
#             plt.savefig(f'{filename}-modrate_{mr}-Lag_{lag}ms.png', dpi=500)
#             plt.clf()
#         else:
#             plt.show()       
# =============================================================================

def resp_overcell(df, cell_idx, saveplot=False):
    '''average response with same parameter, e.g. bandwidth through all cells'''
    ''' for modulation parameter dependece plots'''
    cell_note = pd.read_csv('cell_note_all.csv')
    bd_overcell=[[],[],[],[],[],[],[]]
    cf_overcell=[[],[],[],[],[],[],[],[],[]]
    mod_overcell=[[],[],[],[],[],[]]
    
    for df_loc in cell_idx:
        i = int([i for i,a in enumerate(cell_idx) if a == df_loc][0])
        filename = df['filename'][df_loc]
        version = df['Version'][df_loc]
        cell_data = np.load(f'{filename}_data.npy', allow_pickle=True)
    
        para = cell_data.item().get('para')    
        resp_by_para = cell_data.item().get('resp_by_para')

        resp_bd = resp_by_para['bandwidth']
        resp_cf = resp_by_para['centerfreq']
        resp_mod = resp_by_para['modrate']
        
        cf,band,modrate,_=zip(*para)
        band = sorted(set(band))
        cf = sorted(set(cf))
        modrate = sorted(set(modrate))
        band.remove(0.0)
        modrate.remove(0.0)
        
        i = cell_note.index[cell_note['filename']==filename][0]
        windows = cell_note['window'].loc[i].split(', ')
        
        """Change the index for different region of interest"""
        """0: onset, 1:sustain, 2:offset"""
        window = eval(windows[1])   
        
        '''bandwidth'''
        for idx,res in enumerate(resp_bd):
            res = np.array(res)
            
            def res_crop(arr, window):
                return arr[window[0]:window[1]]
            
            res = np.apply_along_axis(res_crop, 1, res, window)
            res_mean = np.mean(res)
            
            if version==2:
                idx_reduce = idx
                if idx_reduce > 1:
                    idx_reduce+=1
                
                bd_overcell[idx_reduce].append(res_mean)
            else:    
                bd_overcell[idx].append(res_mean)
        
        
        '''center frequency'''
        for idx,res in enumerate(resp_cf):
            res = np.array(res)
            
            def res_crop(arr, window):
                return arr[window[0]:window[1]]
            
            res = np.apply_along_axis(res_crop, 1, res, window)
            res_mean = np.mean(res)

            if version==1 and idx == 0:
                pass
            else:
                idx = idx-1   
                cf_overcell[idx].append(res_mean)
    
    
        '''mod rate'''
        for idx,res in enumerate(resp_mod):
            res = np.array(res)
            
            def res_crop(arr, window):
                return arr[window[0]:window[1]]
            
            res = np.apply_along_axis(res_crop, 1, res, window)
            res_mean = np.mean(res)
            
            if version==2:
                idx_reduce = idx+1                
                mod_overcell[idx_reduce].append(res_mean)
            else:    
                mod_overcell[idx].append(res_mean)
    
    if saveplot:
        mean = [np.mean(res) for res in bd_overcell]
        std = [stats.sem(res) for res in bd_overcell]
        x = np.arange(0,len(mean))
        
        fig, ax = plt.subplots()
        ax.errorbar(x,mean,std, capsize=8, linewidth=3, elinewidth=2)
        ax.set_xlabel('band width (octave)', fontsize=14)
        ax.set_ylabel('membrane potential (mV)', fontsize=14)
        #ax.set_title('offset', fontsize=14)
        ax.set_xticks(np.arange(0,len(x)))
        #ax.set_xticklabels([0.04167, 0.08333, 0.16667, 0.33333, 1.5, 3.0, 7.0])
        ax.set_xticklabels([0.04, 0.08, 0.17, 0.33, 1.5, 3.0, 7.0])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig('bandwidth_sustain.pdf', dpi=500, format='pdf', bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close(fig)
        
        mean = [np.mean(res) for res in cf_overcell]
        std = [stats.sem(res) for res in cf_overcell]
        x = np.arange(0,len(mean))
        
        fig, ax = plt.subplots()
        ax.errorbar(x,mean,std, capsize=5, linewidth=3, elinewidth=2)
        ax.set_xlabel('center frequency (kHz)', fontsize=14)
        ax.set_ylabel('membrane potential (mV)', fontsize=14)
        #ax.set_title('offset', fontsize=14)
        ax.set_xticks(np.arange(0,len(x)))
        ax.set_xticklabels([4.24, 6.0, 8.48, 12.0, 16.97, 24.0, 33.94, 48.0, 67.88], rotation=15)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig('centerfreq_sustain.pdf', dpi=500, format='pdf', bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close(fig)
        
        mean = [np.mean(res) for res in mod_overcell]
        std = [stats.sem(res) for res in mod_overcell]
        x = np.arange(0,len(mean))
        
        fig, ax = plt.subplots()
        ax.errorbar(x,mean,std, capsize=5, linewidth=3, elinewidth=2)
        ax.set_xlabel('mod rate (Hz)', fontsize=14)
        ax.set_ylabel('membrane potential (mV)', fontsize=14)
        #ax.set_title('offset', fontsize=14)
        ax.set_xticks(np.arange(0,len(x)))
        ax.set_xticklabels([1.0, 2.0, 8.0, 16.0, 64.0, 128.0])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig('modrate_sustain.pdf', dpi=500, format='pdf', bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close(fig)
    
    bd = pd.DataFrame(bd_overcell)
    cf = pd.DataFrame(cf_overcell)
    mod = pd.DataFrame(mod_overcell)
    bd.to_csv('bd_offset.csv', index=False)
    cf.to_csv('cf_offset.csv', index=False)
    mod.to_csv('mod_offset.csv', index=False)

    
def stim_resp(i, stim, resp, para, filename, saveplot=False):
    """
    plot superimposed stimulus and response

    Parameters
    ----------
    i : TYPE
        DESCRIPTION.
    stim : TYPE
        DESCRIPTION.
    resp : TYPE
        DESCRIPTION.
    para : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.
    saveplot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    fig, ax1 = plt.subplots()
    ax1.plot()
    
    inst_freq = lsfm_slope.get_instfreq(stim)
    y1 = lsfm_slope.transient_remove(signal.resample(inst_freq, int(len(inst_freq)/8)))
    y1 = [a if a>0 else np.nan for a in y1]
    non_nan = [i for i,a in enumerate(y1) if not math.isnan(a)]
    y1[non_nan[0]-1] = 0
    y1[non_nan[-1]+1] = 0
    x = range(0,len(y1))
    ax1.plot(x,y1, color='red', alpha=0.7)
    ax1.set_title(f'{filename}_#{i}_{para}')
    ax1.set_ylabel('frequency (Hz)')
    ax1.set_xlim(0,len(x))
    
    if len(resp) < 45000:
        ax1.set_xticks(np.linspace(0,len(x),16))
        ax1.set_xticklabels(np.arange(0,16,1)/10, rotation=45)
    else:
        ax1.set_xticks(np.linspace(0,len(x),11))
        ax1.set_xticklabels(np.arange(0,21,2)/10, rotation=45)
    ax1.set_xlabel('time (sec)')
    
    
    ax2 = ax1.twinx()
    y2 = TFTool.butter(resp, 3, 2000, 'lowpass', 25000)
    y2 = lsfm_slope.baseline(y2)
    ax2.plot(x,y2, color='k')
    ax2.set_ylabel('membrane potential (mV)')
    if saveplot:
        plt.savefig(f'{filename}_stim-resp_{i}.pdf', dpi=500, format='pdf', bbox_inches='tight')
        plt.savefig(f'{filename}_stim-resp_{i}.png', dpi=500, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
    plt.close(fig)


def resp_bf_or_not(stim, resp, para, bf, bandwidth=None):
    """
    Devide response by whether stimulus ever crossed best frequency or not.
    If bandwidth is specified, crossing tuning range will be used instead of bf.

    Parameters
    ----------
    resp : TYPE
        DESCRIPTION.
    para : TYPE
        DESCRIPTION.
    bf : float
        Best frequency in Hz.

    Returns
    -------
    resp_bf_in : TYPE
        resp with stimulus crossed bf.
    resp_bf_ex : TYPE
        resp without stimulus ever crossing bf.
    para_bf_in : TYPE
        para for resp_bf_in.
    para_bf_ex : TYPE
        para for resp_bf_ex.

    """
    stim_inbf, stim_exbf = [],[]
    resp_inbf, resp_exbf = [],[]
    para_inbf, para_exbf = [],[]
    index_inbf, index_exbf = [],[]
    
    if bandwidth:
        bf_max = bandwidth[0]
        bf_min = bandwidth[1]
        
        for i,(s,r,p) in enumerate(zip(stim, resp, para)):
            '''calculation frequency band using para'''
            freq_max = p[0]*1000 * (2**(p[1]/2))
            freq_min = p[0]*1000 / (2**(p[1]/2))
            if (bf_max > freq_min and bf_max < freq_max) or (bf_min > freq_min and bf_min < freq_max) or (bf > freq_min and bf < freq_max):
                stim_inbf.append(s)
                resp_inbf.append(r)
                para_inbf.append(p)
                index_inbf.append(i)
            else:
                stim_exbf.append(s)
                resp_exbf.append(r)
                para_exbf.append(p)
                index_exbf.append(i)
    
    else:
    
        for i,(s,r,p) in enumerate(zip(stim, resp, para)):
            '''calculation frequency band using para'''
            freq_max = p[0]*1000 * (2**(p[1]/2))
            freq_min = p[0]*1000 / (2**(p[1]/2))
            if bf > freq_min and bf < freq_max:
                stim_inbf.append(s)
                resp_inbf.append(r)
                para_inbf.append(p)
                index_inbf.append(i)
            else:
                stim_exbf.append(s)
                resp_exbf.append(r)
                para_exbf.append(p)
                index_exbf.append(i)
    
    return stim_inbf, stim_exbf, resp_inbf, resp_exbf, para_inbf, para_exbf, index_inbf, index_exbf


# =============================================================================
# def resp_bfband_or_not(resp, para, bf, bandwidth):
#     """
#     Devide response by whether stimulus ever crossed best frequency or not.
# 
#     Parameters
#     ----------
#     resp : TYPE
#         DESCRIPTION.
#     para : TYPE
#         DESCRIPTION.
#     bf : float
#         Best frequency in Hz.
# 
#     Returns
#     -------
#     resp_bf_in : TYPE
#         resp with stimulus crossed bf.
#     resp_bf_ex : TYPE
#         resp without stimulus ever crossing bf.
#     para_bf_in : TYPE
#         para for resp_bf_in.
#     para_bf_ex : TYPE
#         para for resp_bf_ex.
# 
#     """
#     bf_max = bf*2**(bandwidth/2)
#     bf_min = bf/2**(bandwidth/2)
#     resp_inbf, resp_exbf = [],[]
#     para_inbf, para_exbf = [],[]
#     index_inbf, index_exbf = [],[]
#     
#     for i,p in enumerate(para):
#         '''calculation frequency band using para'''
#         freq_max = p[0]*1000 * (2**(p[1]/2))
#         freq_min = p[0]*1000 / (2**(p[1]/2))
#         if (bf_max > freq_min and bf_max < freq_max) or (bf_min > freq_min and bf_min < freq_max) or (bf > freq_min and bf < freq_max):
#             resp_inbf.append(resp[i])
#             para_inbf.append(p)
#             index_inbf.append(i)
#         else:
#             resp_exbf.append(resp[i])
#             para_exbf.append(p)
#             index_exbf.append(i)
#     
#     return resp_inbf, resp_exbf, para_inbf, para_exbf, index_inbf, index_exbf
# =============================================================================


def best_lags():
    """
    Plot best lags from resp_at_freq.

    Returns
    -------
    None.

    """
    cell_note = pd.read_csv('cell_note_all.csv')
    bf = cell_note['best frequency']
    lag_all = cell_note['best_lag_all']
    lag_first = cell_note['best_lag_bf']
    bf_scale = [i/1000 for i in bf]
    
    
    fig, ax1 = plt.subplots(figsize=(4,6))
    im = ax1.scatter(np.zeros(len(lag_all)), lag_all, zorder=10, c=bf_scale, cmap='plasma', s=100)
    im = ax1.scatter(np.ones(len(lag_first)), lag_first, zorder=10, c=bf_scale, cmap='plasma', s=100)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('best frequency (kHz)', fontsize=16)
    cbar.ax.tick_params(axis='y', which='major', labelsize=14)
    for i in range(len(lag_all)):
        ax1.plot([0,1], [lag_all[i], lag_first[i]], c='k', zorder=0)
    plt.locator_params(axis='x', nbins=2)    
    ax1.set_xlim(-0.5,1.5)
    ax1.set_ylabel('lags (ms)', fontsize=16)
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(['All Cross', 'First Cross'])
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    plt.savefig('best_lags.pdf', dpi=500, format='pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close(fig)
    

def seperate_by_para(stim, resp, para, which_parameter=0):
    """
    sepearte stim, resp, para base on lsfm parameter

    Parameters
    ----------
    stim : 2d-array
        array of stimulus time-series.
    resp : 2d-array
        array of response time-series.
    para : list of tuple
        list of parameter tuples.
    which_parameter : int, optional
        Index of parameter category chose to seperate into group. The default is 0.
        0: carrier frequency (cf)
        1: bandwidth (bw)
        2: modulation rate (mr)

    Returns
    -------
    data : dictionay
        'para_type': store which_parameter for referencing
        'parameter': list of parameters in the chosen parameter category 
        'stim': stimuli grouped by each parameter
        'resp': responses grouped by each parameter
        'para': corresponding full-parameters grouped by each parameter

    """
    
    para_idx = which_parameter
    category = set([p[para_idx] for p in para])
    stim_sep, resp_sep, para_sep = [],[],[]
    
    for c in category:
        stim_cat, resp_cat, para_cat = [],[],[]
        for i , p in enumerate(para):
            if p[para_idx] == c:
                stim_cat.append(stim[i])
                resp_cat.append(resp[i])
                para_cat.append(para[i])
        stim_sep.append(stim_cat)
        resp_sep.append(resp_cat)
        para_sep.append(para_cat)
    
    data = {'para_type':para_idx, 'parameter':category, 'stim':stim_sep, 'resp':resp_sep, 'para':para_sep}
    
    return data



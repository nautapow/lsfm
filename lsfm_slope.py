import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal
from scipy import stats
import scipy.io
import TFTool
import pandas as pd
from scipy.signal.windows import dpss
import math

def baseline(resp_iter):    #correct baseline
    return (resp_iter - np.mean(resp_iter[:50*25]))*100

def baseline_zero(resp_iter):   #fix sound onset to zero
    return (resp_iter - resp_iter[50*25])*100

        

"""Plot slope versus frequency with lag"""
def transient_remove(arr):
    """
    Zero non-stimulation part and start and end transient spike caused by filtering or calculation

    Parameters
    ----------
    arr : nd.array
        array for correction.

    Returns
    -------
    arr : nd.array
        corrected array.

    """
    
    arr_crop = arr[15000:195000]
    arr_std = np.std(arr_crop)
    mask = tuple([(arr < min(arr_crop)-arr_std)|(arr > max(arr_crop)+arr_std)])

    arr[mask]=0    
    return arr        

def get_instfreq(stim):
    fs = 200000
    """cwt decimation rate is 800 to 250Hz"""
    b,a = signal.butter(3, 150, btype='low', fs=fs)
    hil = signal.hilbert(stim)
    phase = np.unwrap(np.angle(hil))
    ifreq = np.diff(phase, prepend=0) / (2*np.pi) * fs
    filt_ifreq = signal.filtfilt(b,a,ifreq)
    
    
    return  filt_ifreq
    
def get_stimslope(stim):
    """
    Return frequencies and slopes for single stimulus and response with specified lag.

    Parameters
    ----------
    stim : ARRAY
        single stimulus.
    resp : ARRAY
        correspond response.
    lag : int
        lag in milliseconds.

    Returns
    -------
    list
        [[x:instant frequency], [y:slopes], [z:response with lag]].

    """
    

    inst_freq = transient_remove(get_instfreq(stim))    
    slope = transient_remove(np.diff(inst_freq, prepend=0))
    
    inst_freq_res = signal.resample(inst_freq, int(len(inst_freq)/8))
    slope_res = signal.resample(slope, int(len(slope)/8))
    
    
    """log raw slope for distribution"""
    def scaling(f):    
        if f == 0:
            return 0
        elif f > 0:
            return math.log(f)
        elif f < 0:
            return -1*math.log(-1*f)
        
    slope_res = [scaling(f) for f in slope_res]
    
    
    return inst_freq_res, slope_res


def data_at_lag(inst_freq, slope, resp, lag, **kwargs):
    fs = 25000
    delay_point = int(lag * (fs/1000))
    b,a = signal.butter(1, 1500, btype='low', fs=fs)
    resp = signal.filtfilt(b,a,resp)
    
    if len(resp) == 50000:
        endpoint = 38750
    elif len(resp) == 37500:
        endpoint = 26250
    
    window = kwargs.get('window')
    if window:
        x = inst_freq[window[0]:window[1]]
        y = slope[window[0]:window[1]]
        z = resp[window[0]+delay_point:window[1]+delay_point] 
    else:
        x = inst_freq[1250:endpoint]
        y = slope[1250:endpoint]
        z = resp[1250+delay_point:endpoint+delay_point]
        
    return [x,y,z]

        

def freq_slope_contour(stim, resp, para, lags, binning=None, filename=None, plot=True, saveplot=False, **kwargs):
    """
    Plot lagged membrane potential contour of stimulus slope vs stimulus instant frequency

    Parameters
    ----------
    stim : array_like
        Stimuli.
    resp : array_like
        Responses.
    para : array_like
        Parameters.
    lags : int or list or ndarray 
        Time delay in ms of reponse relative to stimulus.
    binning : [array, array], optional
        [x edges, y edges] for 2D statistic, N edges should be N bins+1. The default is None.
    filename : str, optional
        Filename for storing plot. The default is None.
    plot : bool, optional
        Set True to show plot. The default is True.
    saveplot : bool, optional
        Set True to save plot. The default is False.
    **kwargs : window = (int, int)
        window = (start, end) in datapoint to specify the time window of interest 

    Returns
    -------
    bin_slope_lags : list of ndarray
        list with lags of 2D-array of slope vs frequency

    """       
    
    """index after parameter exclusion"""
    idx=[i for i, a in enumerate(para) if a[2] not in [0.0,16.0,64.0,128.0]]
    
    inst_freqs, slopes, resps = [],[],[] 
    """get instant frequency and slope for each stimulus"""
    window = kwargs.get('window')
    
    for i in idx:
        inst_freq, slope = get_stimslope(stim[i])
        inst_freqs.append(inst_freq)
        slopes.append(slope)
        resps.append(baseline(resp[i]))
    
    bin_slope_lags=[]
    for lag in lags:
        data = [[],[],[]]
        
        for i in range(len(idx)):
            if window:
                data = np.concatenate((data,data_at_lag(inst_freqs[i], slopes[i], resps[i], lag, window=window)), axis=1)
            else:
                data = np.concatenate((data,data_at_lag(inst_freqs[i], slopes[i], resps[i], lag)), axis=1)

        x_edges = [3000,4240,6000,8480,12000,16970,24000,33940,48000,67880,96000]
        y_edges = np.linspace(-20,20,51)
         
        if binning != None:
            ret = stats.binned_statistic_2d(data[0], data[1], data[2], 'mean', bins=binning)
        else:
            ret = stats.binned_statistic_2d(data[0], data[1], data[2], 'mean', bins=[x_edges,y_edges])
        
        if saveplot:
            plot = True
        
        if plot:
            XX, YY = np.meshgrid(x_edges,y_edges)
            #XX, YY = np.meshgrid(ret[1], ret[2])
            fig, ax1 = plt.subplots()
            pcm = ax1.pcolormesh(XX, YY, ret[0].T, cmap='RdBu_r', vmax=20., vmin=-20.)
            #pcm = ax1.pcolormesh(XX, YY, ret[0].T, cmap='RdBu_r', norm=colors.CenteredNorm())
            
            ax1.set_xscale('log')
            
            ax2 = plt.subplot()
            txt = (f'{filename}-Lag:{lag}ms')
            ax2.text(0,1.02, txt, horizontalalignment='left', transform=ax2.transAxes)
            fig.colorbar(pcm, ax=ax1)
            
            if saveplot:
                if window:
                    plt.savefig(f'{filename}_window-{window}_Lag-{lag}ms.png', dpi=500)
                else:
                    plt.savefig(f'{filename}_Lag_{lag}ms.png', dpi=500)
                plt.clf()
                plt.close()
            else:
                plt.show()
                plt.close()
        
        bin_slope_lags.append(ret[0])
    
    return bin_slope_lags

def m_index(slope_lag):
    if slope_lag.ndim == 1:
        pos = slope_lag[25:]
        neg = slope_lag[:25]
    else:
        pos = np.swapaxes(slope_lag, 1, 0)[25:]
        neg = np.swapaxes(slope_lag, 1, 0)[:25]
    neg = neg[::-1]
    
    index = pos - neg
# =============================================================================
#     x_edges = [3000,4240,6000,8480,12000,16970,24000,33940,48000,67880,96000]
#     y_edges = np.linspace(0,20,26)
#     plt.imshow(index, origin='lower', cmap='RdBu_r', extent=(0,20,0,20), norm=colors.CenteredNorm())
#     plt.show()
#     ttest, p_value = stats.ttest_rel(pos.flatten(), neg.flatten(), nan_policy='omit')
# =============================================================================
    
    return index  


def slope_index(slope_lags, best_freq):
    m_mean, m_mean_bf = [],[]
    m_std, m_std_bf = [],[]  
    boot_mean, boot_mean_std,  = [],[]
    boot_mean_bf, boot_mean_bf_std = [],[]
    boot_std, boot_std_std=[],[]
    boot_std_bf, boot_std_bf_std = [],[]
    bf_bin = TFTool.binlocator(best_freq, [3000,4240,6000,8480,12000,16970,24000,33940,48000,67880,96000])
    
    for slag in slope_lags:
# =============================================================================
#         m_mean.append(np.nanmean(s))
#         m_mean_bf.append(np.nanmean(s[bf_bin]))
#         m_std.append(np.nanstd(s))
#         m_std_bf.append(np.nanstd(s[bf_bin]))
# =============================================================================        
        """all frequency"""
        s = slag.flatten()
        res = TFTool.bootstrap(s, np.nanmean, 2000)
        boot_mean.append(res[1])
        boot_mean_std.append(res[2])
        
        res = TFTool.bootstrap(s, np.nanstd, 2000)
        boot_std.append(res[1])
        boot_std_std.append(res[2])
        
        """best frequency"""
        s = list(slag[bf_bin].flatten())
        res = TFTool.bootstrap(s, np.nanmean, 2000)
        boot_mean_bf.append(res[1])
        boot_mean_bf_std.append(res[2])
        
        res = TFTool.bootstrap(s, np.nanstd, 2000)
        boot_std_bf.append(res[1])
        boot_std_bf_std.append(res[2])
        
    diction = {'mean':boot_mean, 'mean_std':boot_mean_std, 'std':boot_std, 'std_std':boot_std_std}
    bf_diction = {'mean':boot_mean_bf, 'mean_std':boot_mean_bf_std, 'std':boot_std_bf, 'std_std':boot_std_bf_std}
        
    return diction, bf_diction


def plot_slope_index(m, m_bf, filename, saveplot=False):
    x = np.arange(0,len(m['mean']))
    y = m['mean']
    err = np.array(m['mean_std'])
    y_bf = m_bf['mean']
    err_bf = np.array(m_bf['mean_std'])

    y2 = m['std']
    err2 = np.array(m['std_std'])
    y2_bf = m_bf['std']
    err2_bf = np.array(m_bf['std_std'])
    
    fig, ax1 = plt.subplots()       
    ax1.plot(x,y, c='red', linewidth=3, label='Mean_All')
    ax1.fill_between(x, y+err, y-err, color='red', alpha=0.3)
    ax1.plot(x,y_bf, c='orange', linewidth=3, label='Mean_Bf')
    ax1.fill_between(x, y_bf+err_bf, y_bf-err_bf, color='orange', alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(x,y2, c='blue', linewidth=3, label='Std_All')
    ax2.fill_between(x, y2+err2, y2-err2, color='blue', alpha=0.3)
    ax2.plot(x,y2_bf, c='green', linewidth=3, label='Std_Bf')
    ax2.fill_between(x, y2_bf+err2_bf, y2_bf-err2_bf, color='green', alpha=0.3)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_title(f'{filename}')
    ax1.set_xticks([0,10,20,30,40,50])
    ax1.set_xticklabels([0,20,40,60,80,100])
    ax1.set_xlim(0,50)
    ax1.set_xlabel('lag ms')
    
    if saveplot:
        plt.savefig(f'{filename}_index.png', dpi=500, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def direction_map(slope_lags):       
    direction_lags = []
    for s in slope_lags:
        if s.ndim == 1:
            pos = s[25:]
            neg = s[:25]
        else:
            pos = np.swapaxes(s, 1, 0)[25:]
            neg = np.swapaxes(s, 1, 0)[:25]
        neg = neg[::-1]   
        
        directional_map = pos - neg
        direction_lags.append(directional_map)
    
    return direction_lags


def direction_index(direction_lags, best_freq):
    boot_mean, boot_mean_std,  = [],[]
    boot_mean_bf, boot_mean_bf_std = [],[]
    boot_std, boot_std_std=[],[]
    boot_std_bf, boot_std_bf_std = [],[]
    bf_bin = TFTool.binlocator(best_freq, [3000,4240,6000,8480,12000,16970,24000,33940,48000,67880,96000])
    
    for dlag in direction_lags:
        """all"""
        s = dlag.flatten()
        res = TFTool.bootstrap(s, np.nanmean, 2000)
        boot_mean.append(res[1])
        boot_mean_std.append(res[2])
        
        res = TFTool.bootstrap(s, np.nanstd, 2000)
        boot_std.append(res[1])
        boot_std_std.append(res[2])
        
        """best frequency"""
        s = list(dlag[bf_bin].flatten())
        res = TFTool.bootstrap(s, np.nanmean, 2000)
        boot_mean_bf.append(res[1])
        boot_mean_bf_std.append(res[2])
        
        res = TFTool.bootstrap(s, np.nanstd, 2000)
        boot_std_bf.append(res[1])
        boot_std_bf_std.append(res[2])
        
    diction = {'mean':boot_mean, 'mean_std':boot_mean_std, 'std':boot_std, 'std_std':boot_std_std}
    bf_diction = {'mean':boot_mean_bf, 'mean_std':boot_mean_bf_std, 'std':boot_std_bf, 'std_std':boot_std_bf_std}
        
    return diction, bf_diction


  
    
def get_bf(resp, para):
    idx = [i for i,a in enumerate(para) if a[2] == 0]
    x,y = [],[]
    for i in idx:
        x.append(para[i][0])
        y.append(np.mean(baseline(resp[i])))
    
    return x[np.array(y).argmax()]
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
import math
import pandas as pd
import lsfm_slope

def best_freq(resp_tune, para):
    """
    Acquiring best frequency by summing depolarized part of each frequency band
    then perform a Guassian fit.

    Parameters
    ----------
    resp_tune : array_like
        resp_on generated from tunning function
    para : list of tuple
        parameter

    Returns
    -------
    dictionary
        key: best_frequnecy, bandwidth, and fit parameters

    """
    
    def sum_above0(arr):
        return sum(i for i in arr if i > 0)
    
    def gauss(x, H, A, x0, sigma):
        return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    def gauss_fit(x, y):
        """
        preform gaussian fit

        Parameters
        ----------
        x : list or array
            x axis value.
        y : list or array
            y axis value.

        Returns
        -------
        popt : array
            return H, A, x0, sigma.
            peak height = H+A
            peak location = x0
            std = sigma
            FWHM = 2.355*sigma

        """
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
        return popt
    
    freq_sum = np.apply_along_axis(sum_above0, 0, resp_tune)
    _, freq, _ = zip(*para)
    freq = sorted(set(freq))
    x = [math.log(f, 2) for f in freq]
    #x = np.arange(0,len(arr))
    popt = gauss_fit(x,freq_sum)
    peak = popt[0]+popt[1]
    bf = 2**popt[2]
    band = abs(2.355*popt[3])
    tone_charact = {'best_frequency': bf, 'bandwidth': band, 'fit': popt, 'resp_sum': freq_sum}
    
    return tone_charact

def center_mass_layer(resp_loud, freq):
    freq_log = [math.log(i, 2) for i in freq]
    massX = freq_log*resp_loud
    mass_sum = np.sum(resp_loud)
    mass_Xsum = np.sum(massX)
    Xm = 2**(mass_Xsum/mass_sum)
    
    return Xm

def center_mass(resp_tune, freq, loud):
    freq_log = [math.log(i, 2) for i in freq]
    Xv, Yv = np.meshgrid(freq_log, loud)
    massX = Xv*resp_tune
    massY = Yv*resp_tune
    mass_sum = np.sum(resp_tune)
    mass_Xsum = np.sum(massX)
    mass_Ysum = np.sum(massY)
    Xm = mass_Xsum/mass_sum
    Ym = mass_Ysum/mass_sum
    Xm = 2**Xm
    
    return Xm, Ym
        
def tuning_range(resp_pos, resp_sum, bf, freq):   
    """
    finding range around center of mass that contains fixed portion of total response

    Parameters
    ----------
    resp_pos : TYPE
        averaged response in resp_mesh, hyperpolarize brought to zero.
    resp_sum : TYPE
        summation of response, separated by loudness.
    bf : TYPE
        best frequency found by center of mass.
    freq : TYPE
        list of frequency.

    Returns
    -------
    freq[left], freq[right] : float
        lower and upper bound of frequency.

    """
    def find_next2bf(bf, freq):
        for i,f in enumerate(freq):
            if f < bf and freq[i+1] > bf:
                return i
                break
    
    left = find_next2bf(bf, freq)
    right = left+1
    resp_freq_sum = resp_pos[left]+resp_pos[right]
    left-=1
    right+=1
    
    def continue_count(side, count):
        if side == 'left':
            if count <= 0:
                count -= 1
            else:
                count = 0
        
        if side == 'right':
            if count >= 0:
                count += 1
            else:
                count = 0
        
        return count
        
    
    while resp_freq_sum < 0.67*resp_sum:
        """find range containing N proportion of total depolarization"""
        count = 0
        
        if left <= 0:
            resp_freq_sum += resp_pos[right]
            right += 1
            
        elif right >= len(freq)-1:
            resp_freq_sum += resp_pos[left]
            left -= 1
            
        else:            
            if count >= math.log(freq[-1]/freq[0],2) and left > 1:
                left -= 1
                count = 0
            elif count <= -1*math.log(freq[-1]/freq[0],2) and right < len(freq)-1:
                right += 1
                count = 0
            
            if resp_pos[left] > resp_pos[right]:
                resp_freq_sum += resp_pos[left]
                left -= 1
                count = continue_count('left', count)
            elif resp_pos[left] < resp_pos[right]:
                resp_freq_sum += resp_pos[right]
                right += 1
                count = continue_count('right', count)
            else:
                resp_freq_sum += resp_pos[left]+resp_pos[right]
                left -= 1
                right += 1
    
    return freq[left], freq[right]


def octave2bf(bf, freq):
    oct_bf = []
    for f in freq:
        oct_bf.append(math.log((f/bf),2))
    
    return oct_bf

def min_index(arr, num):
    arr = np.array(arr)
    
    return np.argmin(abs(arr - num))

def set_hyper2zero(arr):
    mask = arr < 0
    import copy
    arr_pos = copy.deepcopy(arr)
    arr_pos[mask] = 0
    
    return arr_pos

def tuning(resp, para, filename='', plot=True, saveplot=False, data_return=False, **kwargs):
    window = kwargs.get('window')
    
    if window:
        def on_avg(arr, window):
            base = np.mean(arr[:500])
            arr = (arr-base)*100            
            return np.mean(arr[window[0]:window[1]])       
    else:
        window = 'tone_on'
        def on_avg(arr):
            base = np.mean(arr[:500])
            arr = (arr-base)*100            
            return np.mean(arr[500:3000])
    
    def off_avg(arr):
        base = np.mean(arr[:500])
        arr = (arr-base)*100            
        return np.mean(arr[3000:5500])
    
    
    if len(para[0])==3:
        loud, freq, _ = zip(*para)
    elif len(para[0])==2:
        loud, freq = zip(*para)
    else:
        raise ValueError('parameters should be a tuple with a order of (loudness, frequency, (timing))')
    loud = sorted(set(loud))
    freq = sorted(set(freq))

    if 0.0 in loud:
        loud.remove(0.0)
    if 0.0 in freq:
        freq.remove(0.0)
    
    fs=25000
    #resp_filt = TFTool.prefilter(resp, fs)
    resp_mesh = np.reshape(resp, (len(loud),len(freq),-1))
    
    if 90 in loud:    
        resp_mesh = np.delete(resp_mesh, -1, axis=0)
        loud.pop()
    
    resp_on = np.apply_along_axis(on_avg, 2, resp_mesh, window)
    resp_off = np.apply_along_axis(off_avg, 2, resp_mesh)
    
    x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
    y300 = np.linspace(30,80,301)
    
    Nzero = int(300/(len(freq)-1))-1
    zero2D = np.zeros((6,len(freq),Nzero))        
    upsampleX = np.dstack((resp_on, zero2D)).reshape((6,-1))[:,:301]
    
    filt1D = ndimage.gaussian_filter1d(upsampleX, Nzero)
    
    resp_300 = np.swapaxes(filt1D, 0, 1)        
    """swap frequency to the first axis to slice"""
    
    interpXY=[]
    for freq_layer in resp_300:
        interpXY.append(np.interp(y300, loud, freq_layer))
    
    interpXY = np.array(interpXY)
    
    resp_smooth = np.swapaxes(interpXY, 0, 1)   
    resp_pos = set_hyper2zero(resp_smooth)
    
    bf_loud = []
    for i,x in enumerate(resp_pos):
        bf_loud.append(center_mass_layer(x, x300))
    resp_sum = np.sum(resp_pos, axis=1)
    
    tuning_curve = []
    for i in range(len(y300)):        
        tuning_curve.append(tuning_range(resp_pos[i], resp_sum[i], bf_loud[i], x300))
    
    width70 = math.log((tuning_curve[240][1]/tuning_curve[240][0]), 2)
    """240 == 70db after upsampled to 300"""
    
    curve_left, curve_right = [],[]
    for curve in tuning_curve:
        curve_left.append(curve[0])
        curve_right.append(curve[1])
    
    XX, YY = np.meshgrid(x300, y300)
# =============================================================================
#     plt.pcolormesh(XX, YY, resp_smooth, cmap='RdBu_r', norm=colors.CenteredNorm())
#     plt.xscale('log')
#     plt.colorbar()
# =============================================================================
    
    if plot:
        xlabel = [3,6,12,24,48,96]
        xtick = [i * 1000 for i in xlabel]
        ytick = [30,40,50,60,70,80]
    
        fig = plt.figure()
        grid = plt.GridSpec(2, 1, hspace=0.6, height_ratios=[4,1])
        
        ax1 = fig.add_subplot(grid[0])
        #im = plt.pcolormesh(XX, YY, resp_smooth, cmap='RdBu_r', norm=colors.CenteredNorm())
        
        #ax1.add_collection(im)
        
        im = ax1.pcolormesh(XX, YY, resp_smooth, cmap='RdBu_r', norm=colors.CenteredNorm())
        ax1.set_xscale('log')
        ax1.minorticks_off()
        ax1.set_xticks(xtick)
        ax1.set_xticklabels(xlabel)
        ax1.set_yticks(ytick)
        #ax1.set_yticklabels(ylabel)
        ax1.set_title(f'{filename}_{window}')
        ax1.set_xlabel('Frequency (kHz)')
        ax1.set_ylabel('Loudness (dB SPL)')
        
        ax1.scatter(bf_loud, y300, marker='|', c='limegreen', s=5)
        ax1.fill_betweenx(y300, curve_left, curve_right, color='green', alpha=0.25)
# =============================================================================
#         ax1.scatter(curve_left, y300, marker='|', c='blue', s=5)
#         ax1.scatter(curve_right, y300, marker='|', c='blue', s=5)
# =============================================================================
        cax = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.03,ax1.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('mV')
            
        if saveplot:
            if window:
                plt.savefig(f'{filename}_{window}.pdf', dpi=500, format='pdf', bbox_inches='tight')
                plt.savefig(f'{filename}_{window}.png', dpi=500, bbox_inches='tight')
            else:
                plt.savefig(f'{filename}.pdf', dpi=500, format='pdf', bbox_inches='tight')
                plt.savefig(f'{filename}.png', dpi=500, bbox_inches='tight')
            plt.clf()
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
            
    if data_return:
        return x300, y300, resp_smooth, width70, bf_loud
        
        
def tuning_old(resp, para, filename='', saveplot=False, **kwargs):
    """
    Generate tunning map without smoothing.

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
    bf : dictionary
    {'best_frequency': peak location of fit, 'bandwidth': HMFW of fit, 
     'fit':fit parametes from best_freq(), 'resp_sum':sum of deloparization}
    """
    
    window = kwargs.get('window')
    
    if window:
        def on_avg(arr):
            base = np.mean(arr[:500])
            arr = (arr-base)*100            
            return np.mean(arr[window[0]:window[1]])       
    else:
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
    
    def set_hyper2zero(arr):
        mask = arr < 0
        import copy
        arr_pos = copy.deepcopy(arr)
        arr_pos[mask] = 0
        
        return arr_pos
    
    resp_filt = TFTool.pascal_filter(resp_on)
    resp_pos = set_hyper2zero(resp_filt)    
    #bf_x, bf_y = center_mass(resp_pos, freq, loud)
    
    bf_loud = []
    for i,x in enumerate(resp_pos):
        bf_loud.append(center_mass_layer(x, freq))
    resp_sum = np.sum(resp_pos, axis=1)    
    tuning_curve = []

    for i in range(len(loud)):        
        tuning_curve.append(tuning_range(resp_pos[i], resp_sum[i], bf_loud[i], freq))

    curve_left, curve_right = [],[]
    for curve in tuning_curve:
        curve_left.append(curve[0])
        curve_right.append(curve[1])
        
# =============================================================================
#     methods = ['none', 'bicubic', 'spline16',
#            'hamming', 'quadric',
#            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
# =============================================================================
    method='none'
    
    xfreq = sorted(freq[::int((len(freq)-1)/10)])
    xlabel = [i/1000 for i in xfreq]
    ylabel = [int(i) for i in loud]
    Nx = len(xlabel)
    Ny = len(ylabel)
    #xtick = np.arange(0.5,Nx-0.4,1)
    xtick = np.linspace(0,Nx,Nx)
    ytick = np.linspace(0.5,Ny-0.5,Ny)
    
    scaleX = Nx/((math.log(xfreq[-1],2) - math.log(xfreq[0],2)))
    bf_x_scale, bf_y_scale = [],[]
    for y,x in enumerate(bf_loud):    
        bf_x_scale.append((math.log(x,2) - math.log(xfreq[0],2)) * scaleX)
        bf_y_scale.append(y+0.5)
        
    curve_x_scale_left,  curve_x_scale_right=[],[]
    for i in range(len(loud)):
        curve_x_scale_left.append((math.log(curve_left[i],2) - math.log(xfreq[0],2)) * scaleX)
        curve_x_scale_right.append((math.log(curve_right[i],2) - math.log(xfreq[0],2)) * scaleX)
    curve_y_scale = np.arange(0,len(loud)) + 0.5
        
# =============================================================================
#     #x, y from center of mass for entire receptive area
#     bf_x_scale = (math.log(bf_x,2) - math.log(xfreq[0],2)) * scaleX
#     bf_y_scale = (bf_y - ylabel[0])/10 + 0.5
# =============================================================================
    fig = plt.figure()
    grid = plt.GridSpec(2, 1, hspace=0.6, height_ratios=[4,1])
    
    ax1 = fig.add_subplot(grid[0])
    im = plt.imshow(resp_filt, interpolation=method, origin='lower', aspect='auto', 
                    extent=(0,Nx,0,Ny), cmap='RdBu_r', norm=colors.CenteredNorm())
    ax1.add_image(im)    
    ax1.set_xticks(xtick)
    ax1.set_xticklabels(xlabel, rotation=45)
    ax1.set_yticks(ytick)
    ax1.set_yticklabels(ylabel)
    ax1.set_title(f'{filename}_onset')
    ax1.set_xlabel('Frequency (kHz)')
    ax1.set_ylabel('Loudness (dB SPL)')
    
    ax1.scatter(bf_x_scale, bf_y_scale, marker='o', c='limegreen')
    ax1.scatter(curve_x_scale_left, curve_y_scale, marker='x', c='blue')
    ax1.scatter(curve_x_scale_right, curve_y_scale, marker='x', c='blue')
    
# =============================================================================
#     ax2 = fig.add_subplot(grid[1])
# 
#     ax2.plot(x,freq_sum)
#     ax2.plot(x,y)
#     label = [f/1000 for f in freq]
#     ax2.set_xticks(x[::5])
#     ax2.set_xticklabels(label[::5], rotation=45)
#     ax2.axes.get_yaxis().set_visible(False)
#     pos = [ax1.get_position().x0, ax2.get_position().y0, ax1.get_position().width, ax2.get_position().height]
#     ax2.set_position(pos)
# =============================================================================
    
    cax = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.03,ax1.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('mV')
        
    if saveplot:
        if window:
            plt.savefig(f'{filename}_{window}.pdf', dpi=500, format='pdf', bbox_inches='tight')
            plt.savefig(f'{filename}_{window}.png', dpi=500, bbox_inches='tight')
        else:
            plt.savefig(f'{filename}.pdf', dpi=500, format='pdf', bbox_inches='tight')
            plt.savefig(f'{filename}.png', dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    
# =============================================================================
#     method = 'gaussian'
#     xlabel = freq[::int((len(freq)-1)/10)]
#     xlabel = [i/1000 for i in xlabel]
#     ylabel = [int(i) for i in loud]
#     Nx = len(xlabel)
#     Ny = len(ylabel)
#     xtick = np.arange(0.5,Nx-0.4,1)
#     ytick = np.arange(0.5,Ny-0.4,1)
#     
#     fig, ax1 = plt.subplots()
#     im = plt.imshow(resp_off, interpolation=method, origin='lower', extent=(0,Nx,0,Ny), cmap='RdBu_r', norm=colors.CenteredNorm())
#     ax1.add_image(im)
#     ax1.set_xticks(xtick)
#     ax1.set_xticklabels(xlabel, rotation=45)
#     ax1.set_yticks(ytick)
#     ax1.set_yticklabels(ylabel)
#     ax1.set_title(f'{filename}_offset')
#     ax1.set_xlabel('Frequency kHz')
#     ax1.set_ylabel('dB SPL')
#     
#     cax = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.03,ax1.get_position().height])
#     cbar = plt.colorbar(im, cax=cax)
#     cbar.ax.set_ylabel('mV')
#     if saveplot:
#         plt.savefig(f'{filename}_off.pdf', dpi=500, format='pdf', bbox_inches='tight')
#         plt.savefig(f'{filename}_off.png', dpi=500, bbox_inches='tight')
#         plt.clf()
#         plt.close(fig)
#     else:
#         plt.show()
#         plt.close(fig)
# =============================================================================
        
    
def base_adjust(resp_iter, prestim=20, fs=25000):
    """
    Adjust signal baseline and scale

    Parameters
    ----------
    resp_iter : ndarray
        Trace for adjustment.
    prestim : int, optional
        Prestimulation period in ms for baseline correction. The default is 20 ms.
    fs : int, optional
        Sampling rate in Hz. The default is 25000.

    Returns
    -------
    ndarray
        Adjusted trace.

    """
    return (resp_iter - np.mean(resp_iter[:prestim*int(fs/1000)]))*100

def psth(resp, filename, base_adjust=False, x_in_ms=False, saveplot=False, **kwargs):
    
    if base_adjust:
        resp_base = np.apply_along_axis(base_adjust, 1, resp)
    else:
        resp_base = resp
        
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
    ax.set_ylabel('membrane potential (mV)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    if x_in_ms:
        label = np.linspace(-20,380,6)
        ax.set_xticks(np.linspace(0,10000,6),label)
        ax.set_xlabel('time (ms)', fontsize=16)
    else:
        ax.set_xticks([0,500,1500,3000,5000,7000,9000])
        ax.set_xlabel('data point (2500/100ms)')
        
    if saveplot:
        #plt.savefig(f'{filename}_tone-PSTH.pdf', dpi=500, format='pdf', bbox_inches='tight')
        plt.savefig(f'{filename}_tone-PSTH.png', dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
        
    return y


def psth_bf(resp, para, bf, filename, x_in_ms=False, saveplot=False, **kwargs):
    loud, freq, _ = zip(*para)
    loud = sorted(set(loud))
    freq = np.array(sorted(set(freq)))
    
    idx = [i for i,a in enumerate(np.diff(np.sign(freq - bf))) if a > 0][0]
    target_freq = [freq[idx], freq[idx+1]]
    
    target_resp=[]
    for i, p in enumerate(para):
        if p[1] in target_freq:
            target_resp.append(base_adjust(resp[i]))
    
    target_PSTH = np.mean(target_resp, axis=0)
    err = stats.sem(target_resp, axis=0)
    x = np.arange(len(target_PSTH))
    y = target_PSTH
    
    fig, ax = plt.subplots()
    ax.plot(x,target_PSTH)
    ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
    [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.3) for _x in np.arange(0,5100,500)]
    [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [500,3000]]
    #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.3) for _x in np.arange(0,6000,50)]
    #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.7) for _x in np.arange(0,6000,250)]
    ax.set_title(f'{filename}_tone-PSTH-bf')   
    ax.set_xlim(0,10000)
    ax.set_ylabel('membrane potential (mV)')
    
    if x_in_ms:
        label = np.linspace(-20,380,6)
        ax.set_xticks(np.linspace(0,10000,6),label)
        ax.set_xlabel('time (ms)')
    else:
        ax.set_xticks([0,500,1500,3000,5000,7000,9000])
        ax.set_xlabel('data point (2500/100ms)')
        
    if saveplot:
        plt.savefig(f'{filename}_tone-PSTH_bf.png', dpi=500, format='png', bbox_inches='tight')
        plt.clf()
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    
    return y

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

def tone_inst_freq(stim):
    fs=200000    
    hil = signal.hilbert(stim)
    phase = np.unwrap(np.angle(hil))
    return np.diff(phase, prepend=0) / (2*np.pi) * fs

def clear_out_range(arr):
    arr[3000:] = [0]*len(arr[3000:])
    arr[:500] = [0]*len(arr[:500])
    std = np.std(arr[800:2750])
    arr = [0 if a > max(arr[800:2750])+std else a for a in arr]
    arr = [0 if a < min(arr[800:2750])-std else a for a in arr]
      
    return arr

def tone_stim_resp(i, stim, resp, para, filename):
    fig, ax1 = plt.subplots()
    ax1.plot()
    
    inst_freq = tone_inst_freq(stim)
    y1 = clear_out_range(signal.resample(inst_freq, int(len(inst_freq)/8)))
    x = range(0,len(y1))
    ax1.plot(x,y1, color='red', alpha=0.7)
    ax1.set_title(f'{filename}_#{i}_{para}')
    ax1.set_ylabel('frequency (Hz)')
    ax1.set_xlim(0,len(x))
    
    ax1.set_xticks(np.linspace(0,len(x),9))
    ax1.set_xticklabels(np.linspace(0,400,9), rotation=45)
    ax1.set_xlabel('time (ms)')
    
    
    ax2 = ax1.twinx()
    y2 = TFTool.butter(resp, 3, 2000, 'lowpass', 25000)
    y2 = lsfm_slope.baseline(y2)
    ax2.plot(x,y2, color='k')
    ax2.set_ylabel('membrane potential (mV)')
    
    plt.show()
    plt.clf()
    plt.close(fig)
    
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
    

def resp_merge(resp, para, repeat=2):
    """
    Averaging response with same tone pip parameters.
    After LabView ver1.5 tone pip is played with a default of 2 reapeats. 

    Parameters
    ----------
    resp : ndarray
        response.
    para : list of tuple
        parameter in (loudness, frequency, (timing)).

    Returns
    -------
    resp_merge : ndarray
        averaged response.
    para_merge : list of tuple
        merged parameters.

    """
    datapoints = len(resp[0])
    resp_merge = np.mean(np.reshape(resp, (-1,repeat,datapoints)), axis=1)
    _loud, _freq, _ = zip(*para)
    para_merge = list(zip(_loud, _freq))[::repeat]
    
    return resp_merge, para_merge

def significant(resp_mesh, window=(500,3000)):
    def avg(arr):
        base = np.mean(arr[:500])
        arr = (arr-base)*100            
        return np.mean(arr[window[0]:window[1]])
    
    base_plus, base_minus = [],[]
    for min_loud in resp_mesh[0]:
        resp_region = avg(min_loud)
        #resp@tone_period
        if resp_region>=0:
            base_plus.append(resp_region)
        elif resp_region<=0:
            base_minus.append(resp_region)
            
    CI_plus = stats.t.interval(alpha=0.99, df=len(base_plus)-1, loc=np.mean(base_plus), scale=stats.sem(base_plus))
    CI_minus = stats.t.interval(alpha=0.99, df=len(base_minus)-1, loc=np.mean(base_minus), scale=stats.sem(base_minus))
    
    resp_on = np.apply_along_axis(avg, 2, resp_mesh)
    sig_plus = 1*(resp_on>CI_plus[1])
    sig_minus = -1*(resp_on<CI_minus[0])
    sig = sig_plus+sig_minus
    
    return sig

            
    
    
    
    
    
    
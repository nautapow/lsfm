import numpy as np
from nptdms import TdmsFile
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import scipy.io
import pandas as pd


def element(arr, *arg):
    #find the location of element with content arg in an array 
    return [x for x in range(0, len(arr)) if arr[x] == arg]

def fft(arr, fs):
    fs = fs
    fhat = np.fft.fft(arr)
    p = np.abs(fhat)**2
    f = np.abs(np.fft.fftfreq(len(arr))*fs)
    plt.plot(f,p)


def stft(arr, fs, n=100):
    n = n
    f, t, Zxx = signal.stft(arr, fs = fs, nperseg= n)
    plt.pcolormesh(t, f, np.abs(Zxx)**2, vmin=0, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
def morlet(arr, fs, width):
    N = len(arr)
    fs = fs
    t = np.linspace(0, N/fs, N)
    width = width
    freq = np.linspace(1, fs/2, 100)
    widths = width*fs / (2*freq*np.pi)
    
    cwtm = signal.cwt(arr, signal.morlet2, widths, width=width)
    plt.pcolormesh(t, freq, np.abs(cwtm)**2, cmap='viridis', shading='gouraud')
    plt.show()
    
def mat_out(filename, arr):
    filename = str(filename)
    scipy.io.savemat(filename+'_4cwt.mat', {'stim': arr})

def plot(arr, name):
    plt.plot(arr)
    ax = plt.subplot()
    ax.text(0.02,1.03,{name},transform=ax.transAxes,fontsize=13)
    plt.show()
    plt.clf()

def csv_list(path):
    mdir = Path('path')
    #mdir = "Q:\\[Project] 2020 in-vivo patch with behavior animal\\Raw Results\\"
    
    folder = os.listdir(mdir)
    folder.sort()
    
    try:
        df = pd.read_csv('patch_list.csv')
    except:
        df = pd.DataFrame(columns = ['date', '#', 'path', 'type'])
        
        
    frame = []
    
    for i in range(len(folder)):
        if folder[i] not in list(df['date']):
            _fdir = Path(str(mdir) + '/' + folder[i])
            all_files = os.walk(_fdir)
            
            for _,_,files in all_files:
                for file in files:
                    if file.endswith('.tdms') and file.find('Sound') == -1:
                        
                        path = str(_fdir) + '/' + file
                        tdms_meta = TdmsFile.read_metadata(Path(path))
                        rtype = tdms_meta['Settings'].\
                            properties['Sound Configuration.Run Config.Tone Type']
                        n = path.find('_00')
                        fdict = {'date' : folder[i], '#' : str(path[n+1:n+4]), 'path' : 
                                 path, 'type' : rtype}
                        frame.append(fdict)
    
    df = df.append(frame, ignore_index = True)           
    df.to_csv('new_patch_list.csv', index=False)
  


def pmtm(arr, tapers):
    n = tapers.shape[1]
    mtm = []
    for i in range(n):
        mtm.append(np.multiply(arr, tapers[:,i]))
    mtm = np.abs(np.fft.fft(mtm)**2)
    mtm = np.mean(mtm, axis = 0)
    
    return mtm

    
def find_nearest(val, arr):
    if isinstance(val, np.ndarray):
        idx = []
        for i in val:
            idx.append((np.abs(i - arr)).argmin())
   
    else:
        arr = np.asarray(arr)
        idx = (np.abs(val - arr)).argmin()
    
    return idx

    
def para_merge(para, resp, axis=0):
    """
    return mean response with reduced parameter
    
    Parameters
    ----------
    axis: int
        the axis to reduce to, default 0    
        
        0 for modulation rate
        1 for center frequency
        2 for bandwidth
    """
    cf, bd, mod, _ = zip(*para)
    
    if axis == 0:
        obj = mod
        s = 'Modulation Rate'
    elif axis == 1:
        obj = cf
        s = 'Center Frequency'
    elif axis == 2:
        obj = bd
        s = 'Bandwidth'
    else:
        raise KeyError('Please enter the correct axis code')
        
    value_set = sorted(set(obj))
    mean_resp = []
    for value in value_set:
        index = [i for i, a in enumerate(obj) if a == value]
        res=[]
        for i in index:
            res.append(resp[i])
        
        mean_resp.append(np.mean(res, axis=0))
    
    properties = {'axis': axis, 'parameter': s, 'set': value_set}
    return mean_resp, properties


def para_merge2(para, resp, axis=1):
    """
    return mean response with reduced parameter
    
    Parameters
    ----------
    axis: int
        the axis to take average, default 0    
        
        0 for modulation rate
        1 for center frequency
        2 for bandwidth
    """
    cf, bd, mod, _ = zip(*para)
    
    if axis == 0:
        value_set1 = sorted(set(cf))
        value_set2 = sorted(set(bd))
        obj1 = 0
        obj2 = 1
        s1 = 'Center Frequency'
        s2 = 'Bandwidth'
    elif axis == 1:
        value_set1 = sorted(set(mod))
        value_set2 = sorted(set(bd))
        obj1 = 2
        obj2 = 1
        s1 = 'Modulation Rate'
        s2 = 'Bandwidth'
    elif axis == 2:
        value_set1 = sorted(set(mod))
        value_set2 = sorted(set(cf))
        obj1 = 2
        obj2 = 0
        s1 = 'Modulation Rate'
        s2 = 'Center Frequency'
    else:
        raise KeyError('Please enter the correct axis code')
    
    
    mean_resp=[]
    set1,set2 = [],[]
    for value1 in value_set1:
        for value2 in value_set2:
            res=[]
            for idx, par in enumerate(para):
                if par[obj1] == value1 and par[obj2] == value2:
                    res.append(resp[idx])
            
            '''exclude combination with no value'''
            if np.shape(res)[0]==0:
                pass
            else:
                mean_resp.append(np.mean(res, axis=0))
                set1.append(value1)
                set2.append(value2)
    
    properties = {'axis': axis, 'parameter1': s1, 'parameter2': s2, 'set1': set1, 'set2': set2}
    return mean_resp, properties      
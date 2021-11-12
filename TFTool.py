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
    
    cwtm = signal.cwt(a, signal.morlet2, widths, width=width)
    plt.pcolormesh(t, freq, np.abs(cwtm)**2, cmap='viridis', shading='gouraud')
    plt.show()
    
def mat_out(filename, arr):
    filename = str(filename)
    stim = {'stim': arr}
    scipy.io.savemat(filename+'_4cwt.mat', stim)
    
def mat_in(path):
    path = Path(path)
    cwt = scipy.io.loadmat(path)
    return cwt


def csv_list():
    mdir = Path('/Volumes/bcm-pedi-main-neuro-mcginley/Users/cwchiang/in_vivo_patch/')
    #mdir = "Q:\\[Project] 2020 in-vivo patch with behavior animal\\Raw Results\\"
    os.chdir(mdir)
    
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
                        
                        fdict = {'date' : folder[i], '#' : path[85:88], 'path' : 
                                 path, 'type' : rtype}
                        frame.append(fdict)
    
    df = df.append(frame, ignore_index = True)           
    df.to_csv('new_patch_list.csv', index=False)
        
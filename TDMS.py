import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.fft import fft
import numpy as np
from scipy import signal
import os
import pandas as pd


class Tdms():
    
    def __init__(self, path=''):
        self.path = path
        self.S, self.R = [],[]
        self.Para = {}
        self.Sound = []
        self.misc = []
        self.Rdpk = []
        
    def loadtdms(self, path = '', protocol = 1, prep=True, dePeak=True):
        if path != '':
            self.path = path
        elif self.path == '':
            raise ValueError('Please Load a File')
        
        if self.path[-5:] != '.tdms':
            raise ValueError('Please Load File with .tdms')
        elif not os.path.isfile(self.path):
            raise FileNotFoundError('No Such File in the Directory')
        else:
            filename = self.path
            
    
        if filename[-6] == '_':
            str_s = filename[:-5] + 'Sound' + filename[-5:]
        else:
            str_s = filename[:-5] + '_Sound' + filename[-5:]
        
        if os.path.isfile(str_s):
            sound_file = TdmsFile.open(str_s)
            sound = np.array(sound_file.groups()[0].channels()[0])
        
        tdms_file = TdmsFile.open(filename)
        
        #sampling rate = 25kHz; 1 ms = 25 points
        fs = 25000
        sRate = int(fs/1000)

        #load tdms file
        groups = tdms_file['Untitled']
        stim = groups['Sound'][:]
        resp = groups['PatchPrimary'][:]
        trial_startT = groups['AI Start ms'][:]
        stim_startT = groups['Stimulus Start ms'][:]
        stim = np.array(stim)
        stim_startT = stim_startT - trial_startT
        _channel = 'Tone Parameters'
        n_epochs = len(stim_startT)
        
        if prep:
            #0.1Hz high pass butterworth filter to correct baseline drifting
            b,a = signal.butter(1, 0.1, btype='high', fs=25000)
            resp = signal.filtfilt(b,a,resp)
        
        if dePeak:
            #spikes removal
            peaks,_ = signal.find_peaks(resp, prominence=0.2, height=[None, None], rel_height=0.1, width=[0,100])
            base_left = []
            base_right = []

            for peak in peaks:
                    _re = resp[peak-50:peak+200]
                    _re_diff = np.convolve(np.diff(_re), np.ones(10)/10, mode='same') 
                    index = [i for i in range(len(_re_diff)) if np.abs(_re_diff[i] - 0) > 0.001]
                    
                    if index[0] > 40:
                        index[0] = 25
                    if index[-1] < 100:
                        index[-1] = 150
                    
                    base_left.append(peak-50+index[0])
                    base_right.append(peak-50+index[-1])
                    
            m = np.zeros(len(resp), dtype=bool)
            for i in range(len(base_left)):
                m[base_left[i]:base_right[i]] = True
                
            nopeak = resp[:]
            nopeak[m] = np.nan
            nopeak = pd.Series(nopeak)
            nopeak = list(nopeak.interpolate(limit_direction='both', kind='cubic'))
        else:
            nopeak = resp[:]

        '''
        from scipy.optimize import curve_fit
 
        _base = np.mean(resp[:10000])
        
        def func(x,a):
            return a*x+_base
        
        _xdata = np.linspace(0,1,len(resp))
        _popt, _pcov = curve_fit(func, _xdata, resp)
        resp -= func(_xdata, *_popt)
        plt.plot(resp)
        '''
            
        
        #   protocol = type of recording choose in LabView
        #   1 = 15.5 min lsfm
        #   2 = 3.6 min pure tone
        #   3 = 2.5 min SAM
        protocol = protocol

        
        #depend on protocol, load parameters stored in toneparameters
        #sort by stimuli parameters
        if protocol == 1:
            fc = groups[_channel][::3]
            bdwidth = groups[_channel][1::3]
            mod_rate = groups[_channel][2::3]
            
            
            self.Para = sorted(zip(fc, bdwidth, mod_rate, stim_startT), key=lambda x:x[0:3])
            fc, bdwidth, mod_rate, stim_startT = zip(*self.Para)
            stim_startT = np.array(stim_startT)
            #start time ms in tdms is not accurately capture the onset time of stimuli
            #it is approximately 9ms prior to the actual onset time
            #-250ms, +500ms for covering ISI
            stim_startP = stim_startT*sRate + 9*sRate - 250*sRate
                
            #stim_endP = stim_startP + 1500*sRate + 500*sRate
            for i in range(n_epochs): #np.arange(n_epochs):
                x1 = int(stim_startP[i])
                x2 = x1 + 2000*sRate
                self.misc.append(x1)
                if x1 < 0:
                    lst = np.zeros(abs(x1))
                    ss = np.concatenate((lst,stim[:x2]), axis = 0)
                    rr = np.concatenate((lst,resp[:x2]), axis = 0)
                    nop = np.concatenate((lst,nopeak[:x2]), axis = 0)
                    self.S.append(ss)
                    self.R.append(rr)
                    self.Rdpk.append(nop)
                    try:
                        lst = np.zeros(abs(x1)*8)
                        so = np.concatenate((lst,sound[:x2*8]), axis = 0)
                        self.Sound.append(so)
                    except ValueError:
                        print('No Sound File Exist')
                else:
                    self.S.append(stim[x1:x2])
                    self.R.append(resp[x1:x2])
                    self.Rdpk.append(nopeak[x1:x2])
                try:
                    self.Sound.append(sound[x1*8:x2*8])
                except UnboundLocalError:
                    print('No Sound File Exist')
                    break
                
        
        elif protocol == 2:
            freq = groups[_channel][::2]
            loudness = groups[_channel][1::2]
            
            self.Para = sorted(zip(loudness, freq, stim_startT), key=lambda x:x[0:3])
            loudness, freq, stim_startT = zip(*self.Para)
            stim_startT = np.array(stim_startT)
            
            #start time ms in tdms is not accurately capture the onset time of stimuli
            #it is approximately 9ms prior to the actual onset time
            #-250ms, +500ms for covering ISI
            stim_startP = stim_startT*sRate - 20*sRate
                
            #stim_endP = stim_startP + 1500*sRate + 500*sRate
            for i in range(n_epochs): #np.arange(n_epochs):
                x1 = int(stim_startP[i])
                x2 = x1 + 400*sRate
                self.misc.append(x1)
                if x1 < 0:
                    lst = np.zeros(abs(x1))
                    ss = np.concatenate((lst,stim[:x2]), axis = 0)
                    rr = np.concatenate((lst,resp[:x2]), axis = 0)
                    nop = np.concatenate((lst,nopeak[:x2]), axis = 0)
                    self.S.append(ss)
                    self.R.append(rr)
                    self.Rdpk.append(nop)
                    try:
                        lst = np.zeros(abs(x1)*8)
                        so = np.concatenate((lst,sound[:x2*8]), axis = 0)
                        self.Sound.append(so)
                    except ValueError:
                        print('No Sound File Exist')
                else:
                    self.S.append(stim[x1:x2])
                    self.R.append(resp[x1:x2])
                    self.Rdpk.sppend(nopeak[x1:x2])
                try:
                    self.Sound.append(sound[x1*8:x2*8])
                except UnboundLocalError:
                    print('No Sound File Exist')
                    break
                      
        del tdms_file
        self.rawS = stim[:]
        self.rawR = resp[:]
        self.rawRdpk = nopeak[:]

    def get_misc(self):
        return self.misc
       
    def get_stim(self):
        return self.S, self.Para
    
    def get_resp(self):
        return self.R
    
    def get_dir(self):
        return self.path
    
    def get_sound(self):
        return self.Sound
    
    def get_raw(self):
        return self.rawS, self.rawR
    
    def get_dpk(self):
        return self.Rdpk, self.rawRdpk
        

        

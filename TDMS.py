import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.fft import fft
import numpy as np
from scipy import signal
import os
import pandas as pd
from pathlib import Path


class Tdms():
    
    def __init__(self):
        self.S, self.R = [],[]
        self.Para = {}
        self.Sound = []
        self.misc = []
        self.Rdpk = []
        
    def loadtdms(self, path = '', protocol = 0, load_sound = True, base_correct=True, dePeak=True, precise_timing=False):
        """
        load .tdms file
    
        Parameters
        ----------
        protocol : int
            load different cohort, default 0    
            
            0 for lsfm (log sinusoid frequency modulation)
            1 for puretone
            2 for SAM (sinusoid amplitude modulation)
        
        load_sound : bool
            if 'True', load high-resolution sound file
        base_correct : bool
            if 'True', apply 0.1 Hz high-pass butterworth filter 
            to adjust baseline to zero
        dePeak : bool
            if 'True', perform spike elimination to give 
            pure subthreshold activities
        """
        
        
        self.path = Path(path)
        if self.path == '':
            raise ValueError('Please Load File')
        elif self.path.suffix != '.tdms':
            raise ValueError('Please Load File with .tdms')
        elif not self.path.is_file():
            raise FileNotFoundError('No Such File in the Directory')
        else:
            filename = str(self.path)
            
    
        
        
        tdms_file = TdmsFile.open(filename)
        
        #sampling rate = 25kHz; 1 ms = 25 points
        fs = 25000
        sRate = int(fs/1000)

        """load tdms file"""
        groups = tdms_file['Untitled']
        stim = groups['Sound'][:]
        resp = groups['PatchPrimary'][:]
        trial_startT = groups['AI Start ms'][:]
        stim_startT = groups['Stimulus Start ms'][:]
        stim = np.array(stim)
        stim_startT = stim_startT - trial_startT
        _channel = 'Tone Parameters'
        n_epochs = len(stim_startT)
        
        
        """0.1Hz highpass butterworth filter to correct baseline drifting"""
        if base_correct:
            b,a = signal.butter(1, 0.1, btype='high', fs=25000)
            resp = signal.filtfilt(b,a,resp)
        
        
        """remove spilkes for membrain potential anaylsis"""
        if dePeak:
            peaks,_ = signal.find_peaks(resp, prominence=0.2, height=[None, None], rel_height=0.1, width=[0,100])
            base_left = []
            base_right = []
            m = np.zeros(len(resp), dtype=bool)
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
            for i in range(len(base_left)):
                m[base_left[i]:base_right[i]] = True
                
            nopeak = resp[:]
            nopeak[m] = np.nan
            nopeak = pd.Series(nopeak)
            nopeak = list(nopeak.interpolate(limit_direction='both', kind='cubic'))
        else:
            nopeak = resp[:]
            
        
        """load high-resolution sound file if exist"""
        if load_sound or precise_timing:
            filename = str(self.path)
            if filename[-6] == '_':
                sound_path = filename[:-5] + 'Sound' + filename[-5:]
            else:
                sound_path = filename[:-5] + '_Sound' + filename[-5:]
            
            if os.path.isfile(sound_path):
                sound_file = TdmsFile.open(sound_path)
                sound = np.array(sound_file.groups()[0].channels()[0])
            else:
                print('No sound file in the directory')
                pass

        
        '''
        """alternative baseline correction using linear fit"""
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

        
        #depend on protocol, load parameters stored in toneparameters
        #sort by stimuli parameters
        if protocol == 0:
            fc = groups[_channel][::3]
            bdwidth = groups[_channel][1::3]
            mod_rate = groups[_channel][2::3]
            
            
            self.Para = sorted(zip(fc, bdwidth, mod_rate, stim_startT), key=lambda x:x[0:3])
            fc, bdwidth, mod_rate, stim_startT = zip(*self.Para)
            para = self.Para
            stim_startT = np.array(stim_startT)
            #start time ms in tdms is not accurately capture the onset time of stimuli
            #it is approximately 9ms prior to the actual onset time
            #-250ms, +500ms for covering ISI
            stim_startP = stim_startT*sRate + 9*sRate - 50*sRate
                
            #stim_endP = stim_startP + 1500*sRate + 500*sRate
            for i in range(n_epochs):
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

                else:
                    self.S.append(stim[x1:x2])
                    self.R.append(resp[x1:x2])
                    self.Rdpk.append(nopeak[x1:x2])
                    
                if load_sound and os.path.isfile(sound_path):
                    if x1<0:
                        lst = np.zeros(abs(x1)*8)
                        so = np.concatenate((lst,sound[:x2*8]), axis = 0)
                        self.Sound.append(so)
                    else:
                        self.Sound.append(sound[x1*8:x2*8])
                else:
                    sound = self.S


                
        
        elif protocol == 1:
            freq = groups[_channel][::2]
            loudness = groups[_channel][1::2]
            
            para = sorted(zip(loudness, freq, stim_startT), key=lambda x:x[0:3])
            
            """delete all stimuli with frequency lower than 3k Hz"""
            para[:] = [x for x in para if x[1]>=3000]
            loudness, freq, stim_startT = zip(*para)
            
            if precise_timing:
                sound = sound - np.mean(sound)
                hil = signal.hilbert(sound)
                b,a = signal.butter(1, 300, btype='low', fs=200000)
                hil = signal.filtfilt(b,a,np.abs(hil))
                peaks, prop = signal.find_peaks(hil, width=16000)
                x = prop['left_ips']
                stim_startT = np.array(stim_startT)
                times = stim_startT*200
                for idx, time in enumerate(times):
                    a = np.abs(x - time)
                    if a.min() < 5000:
                        i = np.where(a == a.min())[0][0]
                        stim_startT[idx] = x[i]/200
                    
            
            #start time ms in tdms is not accurately capture the onset time of stimuli
            #it is approximately 9ms prior to the actual onset time
            #-250ms, +500ms for covering ISI
            stim_startP = stim_startT*sRate - 20*sRate
                
            #stim_endP = stim_startP + 1500*sRate + 500*sRate
            for i in range(len(para)): #np.arange(n_epochs):
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

                else:
                    self.S.append(stim[x1:x2])
                    self.R.append(resp[x1:x2])
                    self.Rdpk.append(nopeak[x1:x2])
                
                if load_sound and os.path.isfile(sound_path):
                    if x1<0:
                        lst = np.zeros(abs(x1)*8)
                        so = np.concatenate((lst,sound[:x2*8]), axis = 0)
                        self.Sound.append(so)
                    else:
                        self.Sound.append(sound[x1*8:x2*8])
                else:
                    sound = self.S
                      
        
        del tdms_file
        self.Para = para
        self.rawS = sound
        self.rawR = resp
        self.rawRdpk = nopeak
        

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
        

        

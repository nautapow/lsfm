from nptdms import TdmsFile
from scipy.fft import fft
import numpy as np
from scipy import signal
import os
import pandas as pd
from pathlib import Path

"""
Version 1.5
Update for LabView patch_ver1.5.
Fixed error loading pure-tone data by excluding 0.0 dB and 0.0 Hz in parameters
"""



# =============================================================================
# def hilbert(arr):
#     return signal.hilbert(arr)
# =============================================================================

class pickle():
    #create function in main module for multiprocessing to work    
    def hilbert(self, block):
        return signal.hilbert(block)


def multi_hilbert(arr):
    """
    Section array over 50 million points to facilitate hilbert transformation
    as over 100 million drastically decrese the speed of hilbert

    Parameters
    ----------
    arr : ndarray
        Array to perform hilbert transform

    Returns
    -------
    nd.array
        DESCRIPTION.

    """
    
    
    from multiprocessing import Pool
    
    blocks = []
    size = 50000000
    overlap = 2000
    blocks.append([arr[i:i+size] for i in range(0, len(arr), size-overlap)])
    blocks = np.array(blocks, dtype=object).T[:,0]
    
    mh = pickle()
    p = Pool(4)
    outputs = p.map(mh.hilbert, blocks)
    p.close()
    p.join()
    
    trim = []
    for idx, output in enumerate(outputs):
        if idx == 0:
            trim.append(output[:size-overlap//2])
        elif idx == len(outputs)-1:
            trim.append(output[overlap//2:])
        else:
            trim.append(output[overlap//2:size-overlap//2])
    return np.hstack(trim[:])
    


class Tdms_V1():
    """
    class variable
    stim_raw, resp_raw:     non-sectioned stimulus and response
                    stimulus is low-res if load_sound and precise_timing both set to Flase
    stim, resp, para:     sectioned low-resolution stimulus, response, and parameters
    resp_dpk, resp_dpk_raw:  depeaked response, sectioned and non-sectioned raw
    sound:          sectioned high-resolution stimulus
    misc:           peak start location used for sectioning
    path:           file path
    sRate:          sampling rate for low-res stimulus and response
    """
        

    
    def __init__(self):
        self.stim, self.resp, self.sound = [],[],[]
        self.para = {}
        self.misc = []
        self.resp_dpk = []
        
    def loadtdms(self, path='', protocol=0, load_sound=True, base_correct=True, precise_timing=True):
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
            raise ValueError('Please Load File with tdms Format')
        elif not self.path.is_file():
            raise FileNotFoundError('No Such File in the Directory')
        else:
            fdir = str(self.path)
            
    
        
        
        tdms_file = TdmsFile.open(fdir)
        
        #sampling rate = 25kHz; 1 ms = 25 points
        fs = 25000
        sRate = int(fs/1000)

        """load tdms file"""
        groups = tdms_file['Untitled']
        stim_all = groups['Sound'][:]
        resp_all = groups['PatchPrimary'][:]
        trial_startT = groups['AI Start ms'][:]
        stim_startT = groups['Stimulus Start ms'][:]
        stim_all = np.array(stim_all)
        stim_startT = stim_startT - trial_startT
        _channel = 'Tone Parameters'
        self.version = float(tdms_file['Settings'].properties['Software Version'])
        
        self.resp_raw = resp_all
        
        """0.1Hz highpass butterworth filter to correct baseline drifting"""
        if base_correct:
            b,a = signal.butter(1, 0.001, btype='high', fs=25000)
            resp_all = signal.filtfilt(b,a,resp_all)
        
        
        """remove spilkes for membrain potential anaylsis"""
        peaks,_ = signal.find_peaks(resp_all, prominence=0.2, height=[None, None], rel_height=0.1, width=[0,100])
        base_left = []
        base_right = []
        m = np.zeros(len(resp_all), dtype=bool)
        self.peak_loc = peaks
        
        for peak in peaks:
                _re = resp_all[peak-50:peak+200]
                _re_diff = np.convolve(np.diff(_re), np.ones(10)/10, mode='same') 
                index = [i for i in range(len(_re_diff)) if np.abs(_re_diff[i] - 0) > 0.001]
                #boundary for extrime value
                if index[0] > 40:
                    index[0] = 25
                if index[-1] < 100:
                    index[-1] = 150
                
                base_left.append(peak-50+index[0])
                base_right.append(peak-50+index[-1])
        for i in range(len(base_left)):
            m[base_left[i]:base_right[i]] = True
            
        import copy
        nopeak = copy.deepcopy(resp_all)
        nopeak[m] = np.nan
        nopeak = pd.Series(nopeak)
        nopeak = list(nopeak.interpolate(limit_direction='both', kind='cubic'))
            
        
        """load high-resolution sound file if exist"""
        if load_sound or precise_timing:
            fdir = str(self.path)
            if fdir[-6] == '_':
                sound_path = fdir[:-5] + 'Sound' + fdir[-5:]
            else:
                sound_path = fdir[:-5] + '_Sound' + fdir[-5:]
            
            if os.path.isfile(sound_path):
                sound_file = TdmsFile.open(sound_path)
                sound_all = np.array(sound_file.groups()[0].channels()[0])
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
            centfreq = groups[_channel][::3]
            bd = groups[_channel][1::3]
            modrate = groups[_channel][2::3]
            fc, bdwidth, mod_rate, stim_time = [],[],[],[]
            
            for i,f in enumerate(centfreq):
                if f >= 3.0:
                    fc.append(f)
                    bdwidth.append(bd[i])
                    mod_rate.append(modrate[i])
                    stim_time.append(stim_startT[i])
            
            stim_startT = stim_time[:]                    
            
            para = sorted(zip(fc, bdwidth, mod_rate, stim_startT), key=lambda x:x[0:3])
            fc, bdwidth, mod_rate, stim_startT = zip(*para)
            stim_startT = np.array(stim_startT)
            
            if precise_timing:
                _sound = np.array(sound_all - np.mean(sound_all))
                
                hil = np.abs(multi_hilbert(_sound)) - 0.012
                #hil = np.abs(signal.hilbert(_sound)) - 0.012
                b,a = signal.butter(3, 300, btype='low', fs=200000)
                filt = signal.filtfilt(b,a, hil)
                cross0 = np.diff(np.sign(filt)) > 0
                
                #stim_startT in ms, change to data point by times 200 sampling rate
                #windows given 10ms prior and 20ms after the LabView onset timing
                #precision is the index first datapoint crossing threshold
                #counting from 10ms before stim_startT so substract 10ms after switch back to ms by dividing sampling rate
                for idx,time in enumerate(stim_startT):
                    window = cross0[int(time*200-10*200):int(time*200+20*200)]
                    
                    if any(window):
                        precision = min([i for i, x in enumerate(window) if x])/200 - 10
                        stim_startT[idx] = stim_startT[idx] + precision
                    else:
                        stim_startT[idx] = stim_startT[idx] + 9
            
            self.para = zip(fc, bdwidth, mod_rate, stim_startT)
                        
                
            
            stim_startT = np.array(stim_startT)
            #start time ms in tdms is not accurately capture the onset time of stimuli
            #it is approximately 9ms prior to the actual onset time
            #-250ms, +500ms for covering ISI
            stim_startP = stim_startT*sRate - 50*sRate  
            #stim_endP = stim_startP + 1500*sRate + 500*sRate
            for i in range(len(para)):
                x1 = int(stim_startP[i])
                x2 = x1 + 2000*sRate
                self.misc.append(x1)
                if x1 < 0:
                    lst = np.zeros(abs(x1))
                    ss = np.concatenate((lst,stim_all[:x2]), axis = 0)
                    rr = np.concatenate((lst,resp_all[:x2]), axis = 0)
                    nop = np.concatenate((lst,nopeak[:x2]), axis = 0)
                    self.stim.append(ss)
                    self.resp.append(rr)
                    self.resp_dpk.append(nop)

                else:
                    self.stim.append(stim_all[x1:x2])
                    self.resp.append(resp_all[x1:x2])
                    self.resp_dpk.append(nopeak[x1:x2])
                    
                if load_sound and os.path.isfile(sound_path):
                    if x1<0:
                        lst = np.zeros(abs(x1)*8)
                        so = np.concatenate((lst,sound_all[:x2*8]), axis = 0)
                        self.sound.append(so)
                    else:
                        self.sound.append(sound_all[x1*8:x2*8])
                else:
                    self.sound = []
                

                
        
        elif protocol == 1:
            freq = groups[_channel][::2]
            loudness = groups[_channel][1::2]
            
            
            para = sorted(zip(loudness, freq, stim_startT), key=lambda x:x[0:3])
            
            """delete all stimuli with frequency lower than 3k Hz"""
            para[:] = [x for x in para if x[1]>=3000]
            loudness, freq, stim_startT = zip(*para)
            
            if precise_timing:
                _sound = sound_all - np.mean(sound_all)
                hil = signal.hilbert(_sound)
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
            #stim_startP = stim_startT*sRate - 20*sRate
            stim_startP = stim_startT*sRate - 20*sRate #20ms for baseline
            
            for i in range(len(stim_startT)): #np.arange(n_epochs):
                x1 = int(stim_startP[i])
                x2 = x1 + 400*sRate
                self.misc.append(x1)
                if x1 < 0:
                    lst = np.zeros(abs(x1))
                    ss = np.concatenate((lst,stim_all[:x2]), axis = 0)
                    rr = np.concatenate((lst,resp_all[:x2]), axis = 0)
                    nop = np.concatenate((lst,nopeak[:x2]), axis = 0)
                    self.stim.append(ss)
                    self.resp.append(rr)
                    self.resp_dpk.append(nop)

                else:
                    self.stim.append(stim_all[x1:x2])
                    self.resp.append(resp_all[x1:x2])
                    self.resp_dpk.append(nopeak[x1:x2])
                
                if load_sound and os.path.isfile(sound_path):
                    if x1<0:
                        lst = np.zeros(abs(x1)*8)
                        so = np.concatenate((lst,sound_all[:x2*8]), axis = 0)
                        self.sound.append(so)
                    else:
                        self.sound.append(sound_all[x1*8:x2*8])
                else:
                    self.sound = []
                      
        
        del tdms_file
        self.para = para
        self.stim_raw = sound_all
        
        self.resp_dpk_raw = nopeak
        self.sRate = sRate

    def loadsound(self, protocol=0):
        """
        to load cooresponding high-resolution sound file after loading the TDMS file

        Parameters
        ----------
        protocol : TYPE, optional
            0 for lsfm
            1 for pure tone
            The default is 0.

        Raises
        ------
        FileNotFoundError
            no sound file match the loaded TDMS file

        Returns
        -------
        list
            list of array

        """
        fdir = str(self.path)
        if fdir[-6] == '_':
            sound_path = fdir[:-5] + 'Sound' + fdir[-5:]
        else:
            sound_path = fdir[:-5] + '_Sound' + fdir[-5:]
        
        if os.path.isfile(sound_path):
            sound_file = TdmsFile.open(sound_path)
            sound_all = np.array(sound_file.groups()[0].channels()[0])
            self.stim_raw = sound_all
        else:
            raise FileNotFoundError('No sound file in the directory')
      
        self.Sound = []
        
        for x1 in self.misc:
            if protocol == 0:
                x2 = x1 + 2000*self.sRate
            elif protocol == 1:
                x2 = x1 + 400*self.sRate
            
            if x1<0:
                lst = np.zeros(abs(x1)*8)
                so = np.concatenate((lst,sound_all[:x2*8]), axis = 0)
                self.sound.append(so)
            else:
                self.sound.append(sound_all[x1*8:x2*8])
        
        return self.sound
            
    def cut(self, arr, protocol=0):
        cutted = []        
        for x1 in self.misc:
            if protocol == 0:
                x2 = x1 + 2000*self.sRate
            elif protocol == 1:
                x2 = x1 + 400*self.sRate
            
            if x1<0:
                lst = np.zeros(abs(x1)*8)
                _c = np.concatenate((lst,arr[:x2*8]), axis = 0)
                cutted.append(_c)
            else:
                cutted.append(arr[x1*8:x2*8])
        
        return cutted
        
    
class Tdms_V2():
    
    def __init__(self):
        self.stim, self.resp, self.sound = [],[],[]
        self.para = {}
        self.misc = []
        self.resp_dpk = []
        
    def loadtdms(self, path = '', protocol=0, load_sound = True, base_correct=True, dePeak=True):
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
            fdir = str(self.path)
        
        
        tdms_file = TdmsFile.open(fdir)
        
        #sampling rate = 25kHz; 1 ms = 25 points
        fs = 25000
        sRate = int(fs/1000)

        """load tdms file"""
        groups = tdms_file['Untitled']
        stim_all = groups['Sound'][:]
        resp_all = groups['PatchPrimary'][:]
        trial_startT = groups['AI Start ms'][:]
        stim_startT = groups['Stimulus Start ms'][:]
        timing = groups['StimStart'][:]
        stim_all = np.array(stim_all)
        stim_startT = stim_startT - trial_startT
        _channel = 'Tone Parameters'
        self.version = float(tdms_file['Settings'].properties['Software Version'])
        
        self.resp_raw = resp_all
        
        """0.1Hz highpass butterworth filter to correct baseline drifting"""
        if base_correct:
            b,a = signal.butter(1, 0.1, btype='high', fs=25000)
            resp_all = signal.filtfilt(b,a,resp_all)
        
        
        """remove spilkes for membrain potential anaylsis"""
        peaks,_ = signal.find_peaks(resp_all, prominence=0.2, height=[None, None], rel_height=0.1, width=[0,100])
        base_left = []
        base_right = []
        m = np.zeros(len(resp_all), dtype=bool)
        for peak in peaks:
                _re = resp_all[peak-50:peak+200]
                _re_diff = np.convolve(np.diff(_re), np.ones(10)/10, mode='same') 
                index = [i for i in range(len(_re_diff)) if np.abs(_re_diff[i] - 0) > 0.001]
                #boundary for extrime value
                if index:
                    if index[0] > 40:
                        index[0] = 25
                    if index[-1] < 100:
                        index[-1] = 150
                    base_left.append(peak-50+index[0])
                    base_right.append(peak-50+index[-1])
                else:
                    pass
        for i in range(len(base_left)):
            m[base_left[i]:base_right[i]] = True
            
        import copy
        nopeak = copy.deepcopy(resp_all)
        nopeak[m] = np.nan
        nopeak = pd.Series(nopeak)
        nopeak = list(nopeak.interpolate(limit_direction='both', kind='cubic'))

            
        
        """load high-resolution sound file if exist"""
        if load_sound:
            fdir = str(self.path)
            if fdir[-6] == '_':
                sound_path = fdir[:-5] + 'Sound' + fdir[-5:]
            else:
                sound_path = fdir[:-5] + '_Sound' + fdir[-5:]
            
            if os.path.isfile(sound_path):
                sound_file = TdmsFile.open(sound_path)
                sound_all = np.array(sound_file.groups()[0].channels()[0])
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
            cross0 = np.diff(np.sign(timing-2)) > 0
            #stim_time = [i for i, a in enumerate(np.diff(timing, prepend=0)) if a > 3]
            stim_time = [i for i,a in enumerate(cross0) if a]
                 
            _para_sort = sorted(zip(fc, bdwidth, mod_rate, stim_time), key=lambda x:x[0:3])
            fc, bdwidth, mod_rate, stim_time = zip(*_para_sort)
            stim_time = np.array(stim_time)
            para = {'fc':fc, 'bdwidth':bdwidth, 'mod_rate':mod_rate, 'stim_time':stim_time}
            
            
            stim_startP = stim_time - 50*sRate
            #stim_endP = stim_startP + 1500*sRate + 500*sRate
            for i in range(len(stim_time)):
                x1 = int(stim_startP[i])
                x2 = x1 + 1500*sRate
                self.misc.append(x1)
                if x1 < 0:
                    lst = np.zeros(abs(x1))
                    ss = np.concatenate((lst,stim_all[:x2]), axis = 0)
                    rr = np.concatenate((lst,resp_all[:x2]), axis = 0)
                    nop = np.concatenate((lst,nopeak[:x2]), axis = 0)
                    self.stim.append(ss)
                    self.resp.append(rr)
                    self.resp_dpk.append(nop)

                else:
                    self.stim.append(stim_all[x1:x2])
                    self.resp.append(resp_all[x1:x2])
                    self.resp_dpk.append(nopeak[x1:x2])
                    
                if load_sound and os.path.isfile(sound_path):
                    if x1<0:
                        lst = np.zeros(abs(x1)*8)
                        so = np.concatenate((lst,sound_all[:x2*8]), axis = 0)
                        self.sound.append(so)
                    else:
                        self.sound.append(sound_all[x1*8:x2*8])
                else:
                    self.sound = []
                


                
        
        elif protocol == 1:
            freq = groups[_channel][::2]
            loudness = groups[_channel][1::2]
            cross0 = np.diff(np.sign(timing-2)) > 0
            #stim_time = [i for i, a in enumerate(np.diff(timing, prepend=0)) if a > 3]
            stim_time = [i for i,a in enumerate(cross0) if a]
            
            if self.version==1.5:
                freq = np.array([int(x) for x in freq if x != 0.0])
                loudness = np.array([int(x) for x in loudness if x != 0.0])
            
            _para_sort = sorted(zip(loudness, freq, stim_time), key=lambda x:x[0:2])
            loudness, freq, stim_time = zip(*_para_sort)
            stim_time = np.array(stim_time)
            para = {'loudness':loudness, 'freq':freq, 'stim_time':stim_time}
            para_zip = list(zip(loudness, freq, stim_time))

            
            #start time ms in tdms is not accurately capture the onset time of stimuli
            #it is approximately 9ms prior to the actual onset time
            #-250ms, +500ms for covering ISI
            #stim_startP = stim_startT*sRate - 20*sRate
            stim_startP = stim_time - 20*sRate
            
            for i in range(len(stim_time)):
                x1 = int(stim_startP[i])
                x2 = x1 + 400*sRate
                self.misc.append(x1)
                if x1 < 0:
                    lst = np.zeros(abs(x1))
                    ss = np.concatenate((lst,stim_all[:x2]), axis = 0)
                    rr = np.concatenate((lst,resp_all[:x2]), axis = 0)
                    nop = np.concatenate((lst,nopeak[:x2]), axis = 0)
                    self.stim.append(ss)
                    self.resp.append(rr)
                    self.resp_dpk.append(nop)

                else:
                    self.stim.append(stim_all[x1:x2])
                    self.resp.append(resp_all[x1:x2])
                    self.resp_dpk.append(nopeak[x1:x2])
                
                if load_sound and os.path.isfile(sound_path):
                    if x1<0:
                        lst = np.zeros(abs(x1)*8)
                        so = np.concatenate((lst,sound_all[:x2*8]), axis = 0)
                        self.sound.append(so)
                    else:
                        self.sound.append(sound_all[x1*8:x2*8])
                else:
                    self.sound = []
                      
        
        del tdms_file
        self.para = para_zip
        self.stim_raw = sound_all
        
        self.resp_dpk_raw = nopeak
        self.sRate = sRate

    def loadsound(self, protocol=0):
        """
        to load cooresponding high-resolution sound file after loading the TDMS file

        Parameters
        ----------
        protocol : TYPE, optional
            0 for lsfm
            1 for pure tone
            The default is 0.

        Raises
        ------
        FileNotFoundError
            no sound file match the loaded TDMS file

        Returns
        -------
        list
            list of array

        """
        fdir = str(self.path)
        if fdir[-6] == '_':
            sound_path = fdir[:-5] + 'Sound' + fdir[-5:]
        else:
            sound_path = fdir[:-5] + '_Sound' + fdir[-5:]
        
        if os.path.isfile(sound_path):
            sound_file = TdmsFile.open(sound_path)
            sound_all = np.array(sound_file.groups()[0].channels()[0])
            self.stim_raw = sound_all
        else:
            raise FileNotFoundError('No sound file in the directory')
      
        self.dound = []
        
        for x1 in self.misc:
            if protocol == 0:
                x2 = x1 + 1500*self.sRate
            elif protocol == 1:
                x2 = x1 + 400*self.sRate
            
            if x1<0:
                lst = np.zeros(abs(x1)*8)
                so = np.concatenate((lst,sound_all[:x2*8]), axis = 0)
                self.sound.append(so)
            else:
                self.sound.append(sound_all[x1*8:x2*8])
        
        return self.sound
            
    def cut(self, arr, protocol=0):
        cutted = []        
        for x1 in self.misc:
            if protocol == 0:
                x2 = x1 + 1500*self.sRate
            elif protocol == 1:
                x2 = x1 + 400*self.sRate
            
            if x1<0:
                lst = np.zeros(abs(x1)*8)
                _c = np.concatenate((lst,arr[:x2*8]), axis = 0)
                cutted.append(_c)
            else:
                cutted.append(arr[x1*8:x2*8])
        
        return cutted
        

        

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.image import NonUniformImage
from matplotlib.widgets import Button, RadioButtons
import matplotlib
from pathlib import Path
from scipy import signal
from scipy import stats
from scipy.optimize import curve_fit
from scipy import ndimage
import TFTool
import math
import pandas as pd
import lsfm_slope


def get_info(list_of_interests):
    if os.path.isfile('tone_cell_note.xlsx'):
        df_info = pd.read_excel('tone_cell_note.xlsx')
        for i in list_of_interests:
            if i not in list(df_info['index']):
                print(f'index {i} not in record\n')
        
        print('lsfm_cell_note.xlsx will be renamed to update new entry')
        try:
            os.rename('tone_cell_note.xlsx', 'tone_cell_note_old.xlsx')
        except FileExistsError:
            os.remove('tone_cell_note_old.xlsx')
            os.rename('tone_cell_note.xlsx', 'tone_cell_note_old.xlsx')
               
    df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
    
    filenames, index, version, mouseID, site, bf, bandwidth, band_left, band_right=[],[],[],[],[],[],[],[],[]
    for idx in list_of_interests:
        filename = df['filename'][idx]
        filenames.append(filename)
        index.append(idx)
        version.append(df['Py_version'][idx])
        mouse = df['mouse_id'][idx]
        mouseID.append(mouse)
        site.append(df['patch_site'][idx])
        
        data = np.load(f'{filename}.npy', allow_pickle=True)
        resp = data.item()['resp']
        para = data.item()['para']
   
        from tone import PureTone  
        tone1 = PureTone(resp, para, mouse, filename)
        tone1.get_bf()
        tone1.get_bandwidth()
        bf.append(tone1.bf)
        bandwidth.append(tone1.bandwidth)
        band_left.append(tone1.band_left)
        band_right.append(tone1.band_right)
        
    data = pd.DataFrame({'filename':filenames, 'index':index, 'version':version, 'mouseID':mouseID, 'patch_site':site,
            'best_frequency':bf, 'bandwidth':bandwidth, 'band_left':band_left, 'band_right':band_right})
    data.to_excel('tone_cell_note.xlsx')


def merge_repeat(resp, para, repeat=2):
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

def set_hyper2zero(arr):
    mask = np.array(arr) < 0
    import copy
    arr_pos = copy.deepcopy(arr)
    arr_pos[mask] = 0
    
    return arr_pos

def pascal_filter(arr):
    """
    Apply Pascal filter along axis=0 on a (3, N) array.
    Returns a 1D array of shape (N,) where each element is computed as:
    O = (1 * arr[0, :] + 2 * arr[1, :] + 1 * arr[2, :]) / 4
    """
    if arr.shape[0] != 3:
        raise ValueError("Input array must have shape (3, N)")
    
    weights = np.array([1, 2, 1])[:, None]  # shape (3, 1) for broadcasting
    weighted_sum = np.sum(arr * weights, axis=0)
    return weighted_sum / 4

def pascal_filter_level(data):
    """
    Apply Pascal-weighted smoothing along axis 0 of a 2D array.
    
    For first row:      (2*row0 +   row1) / 3
    For middle rows:    (row[i-1] + 2*row[i] + row[i+1]) / 4
    For last row:       (row[-2] + 2*row[-1]) / 3
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    M, N = data.shape
    if M < 2:
        raise ValueError("Input must have at least 2 rows.")

    smoothed = np.empty_like(data, dtype=float)

    # First row
    smoothed[0] = (2 * data[0] + data[1]) / 3

    # Middle rows
    for i in range(1, M - 1):
        smoothed[i] = (data[i - 1] + 2 * data[i] + data[i + 1]) / 4

    # Last row
    smoothed[-1] = (data[-2] + 2 * data[-1]) / 3

    return smoothed


def pascal_smooth_loudness(data):
    """
    Apply Pascal smoothing (1, 2, 1) along the loudness axis (axis=0) for each frequency.

    Parameters:
        data (np.ndarray): A 2D array of shape (6, 26), where axis 0 is loudness (from 30 to 80 dB), and axis 1 is frequency.

    Returns:
        np.ndarray: Smoothed data of same shape (6, 26)
    """
    if data.shape[0] != 6:
        raise ValueError("Expected data to have 6 loudness levels along axis 0.")

    smoothed = np.zeros_like(data, dtype=float)

    for j in range(data.shape[1]):  # loop over each frequency
        for i in range(data.shape[0]):  # loop over each loudness
            vals = []
            weights = []

            if i > 0:
                vals.append(data[i-1, j])
                weights.append(1)
            vals.append(data[i, j])
            weights.append(2)
            if i < data.shape[0] - 1:
                vals.append(data[i+1, j])
                weights.append(1)

            weights = np.array(weights)
            vals = np.array(vals)
            smoothed[i, j] = np.sum(vals * weights) / np.sum(weights)

    return smoothed

def pascal_smooth_loudness_large(data):
    """
    Apply Pascal smoothing [1, 4, 6, 4, 1] along the loudness axis (axis=0) for each frequency.

    Parameters:
        data (np.ndarray): A 2D array of shape (6, 26), where axis 0 is loudness, and axis 1 is frequency.

    Returns:
        np.ndarray: Smoothed data of same shape (6, 26)
    """
    if data.shape[0] != 6:
        raise ValueError("Expected data to have 6 loudness levels along axis 0.")

    kernel = np.array([1, 4, 6, 4, 1])
    half_k = len(kernel) // 2  # = 2
    smoothed = np.zeros_like(data, dtype=float)

    for j in range(data.shape[1]):  # for each frequency
        for i in range(data.shape[0]):  # for each loudness
            vals = []
            weights = []

            for k in range(-half_k, half_k + 1):
                idx = i + k
                if 0 <= idx < data.shape[0]:
                    vals.append(data[idx, j])
                    weights.append(kernel[k + half_k])

            vals = np.array(vals)
            weights = np.array(weights)
            smoothed[i, j] = np.sum(vals * weights) / np.sum(weights)

    return smoothed


def pascal_filter_convolve(data, weights=[1,4,6,4,1]):
    """
    Apply Pascal-weighted smoothing along axis 0 of a 2D array using convolution.

    Parameters:
        data (2D array): Input array.
        weights (list): Pascal weights (default [1, 2, 1]).

    Returns:
        2D array: Smoothed result.
    """
    from scipy.ndimage import convolve1d
    
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    if len(weights) % 2 == 0:
        raise ValueError("Weights must have an odd length for symmetric smoothing.")

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()  # normalize weights

    # Apply convolution along axis 0 with reflect padding to handle boundaries
    smoothed = convolve1d(data, weights, axis=0, mode='nearest')

    return smoothed

def pascal_filter_1D(arr):
    kernel = np.array([1, 4, 6, 4, 1])
    radius = len(kernel) // 2
    n = len(arr)
    result = np.zeros_like(arr, dtype=float)

    for i in range(n):
        vals = []
        weights = []

        for offset in range(-radius, radius + 1):
            idx = i + offset
            if 0 <= idx < n:
                weight = kernel[offset + radius]
                vals.append(weight * arr[idx])
                weights.append(weight)

        result[i] = sum(vals) / sum(weights)

    return result

def upscale_fft(data, num_points=301):
    import scipy.fft as fft

    N = len(data)
    data_fft = fft.fft(data)
    padded_fft = np.concatenate([
        data_fft[:N//2],
        np.zeros(num_points - N),
        data_fft[N//2:]
    ])
    result = fft.ifft(padded_fft) * (num_points / N)
    
    return np.real(result)

def find_which_bin(bf, freq_bin):
    for i,(f1,f2) in enumerate(zip(freq_bin[:-1], freq_bin[1:])):
        if f1 <= bf and f2 > bf:
            return i
            break
        elif (i==0 and f1>=bf) or (i==len(freq_bin[:-1])-1 and f2<=bf):
            return i
            break


class PureTone:
    def __init__(self, resp, para, mouseID, filename):
        """
        

        Parameters
        ----------
        resp : 2darray
            responses to puretones
        para : list
            list of tuple as (loudness, frequency, (stim time))
        filename : str
            current working file

        Returns
        -------
        resp_mesh: 3darray
        resp structrued as (sound_level, frequency, raw trace)
        resp_on_mesh: 2darray
        response averaged within onset window, sturctured as (sound_level, frequency)

        """
        self.resp_raw = resp
        self.para_raw = para
        self.filename = filename
        self.mouseID = mouseID
        
        resp = np.array([100*(r-np.mean(r[250:500])) for r in self.resp_raw])
        
        if len(self.para_raw[0])==3:
            loud, freq, _ = zip(*self.para_raw)
        elif len(self.para_raw[0])==2:
            loud, freq = zip(*self.para_raw)
        else:
            raise ValueError('parameters should be a tuple with a order of (loudness, frequency, (timing))')
        loud = sorted(set(loud))
        freq = sorted(set(freq))
        
        if 0.0 in loud:
            loud.remove(0.0)
        if 0.0 in freq:
            freq.remove(0.0)
            
        self.loud = loud
        self.freq = freq
        
        """NOTE! self.resp and self.para is baseline corrected and repeat averaged."""
        resp_merge, para_merge = merge_repeat(resp, para, repeat=2)
        self.resp = resp_merge
        self.para = para_merge
        
        self.psth = np.mean(self.resp, axis=0)
        self.resp_mesh = np.reshape(self.resp, (len(self.loud), len(self.freq), -1))
        self.resp_on_mesh = np.mean(self.resp_mesh[:,:,500:3000], axis=2)
        self.resp_off_mesh = np.mean(self.resp_mesh[:,:,3000:5500], axis=2)
        self._bf=None
        self._bandwidth=None
        
    @property
    def bf(self):
        """Getter that raises an error if value is not set."""
        if self._bf is None:
            raise AttributeError(f"use attribute \"get_bf(\'on\')\" to acquire best frequency")
        return self._bf
    
    @bf.setter
    def bf(self, value):
        self._bf = value
    
    @property
    def bandwidth(self):
        """Getter that raises an error if value is not set."""
        if self._bandwidth is None:
            raise AttributeError("use attribute \"get_bandwidth(\'on\')\" to acquire bandwidth")
        return self._bandwidth
    
    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value
    
    
# =============================================================================
#     def get_level(self, upscale=False):
#         #bf
#         working_resp = self.resp_on_mesh
#         x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
#         Nzero = int(300/(len(self.freq)-1))-1
#         zero2D = np.zeros((6,len(self.freq),Nzero))        
#         upsampleX = np.dstack((working_resp, zero2D)).reshape((6,-1))[:,:301]
#         
#         if not upscale:
#             bf_level, resp_pos_level=[],[]
#             for level in range(6):
#                 filt1D = ndimage.gaussian_filter1d(upsampleX, Nzero)[level]
#                 resp_pos = set_hyper2zero(filt1D)
#                 
#                 """bf with center of mass"""
#                 freq_log = [math.log(i, 2) for i in x300]
#                 massX = freq_log*resp_pos
#                 mass_sum = np.sum(resp_pos)
#                 mass_Xsum = np.sum(massX)
#                 center_mass = (2**(mass_Xsum/mass_sum))
#                 if self.check_authantic(center_mass, resp_pos):
#                     center_mass = x300[np.argmax(resp_pos)]
#                 
#                 bf_level.append(center_mass)
#                 resp_pos_level.append(resp_pos)
#         else:
#             y300 = np.linspace(30,80,301)
#             filt1D = ndimage.gaussian_filter1d(upsampleX, Nzero)
#             resp_300 = np.swapaxes(filt1D, 0, 1)        
# 
#             interpXY=[]
#             for freq_layer in resp_300:
#                 interpXY.append(np.interp(y300, self.loud, freq_layer))
#             
#             interpXY = np.array(interpXY)
#             
#             resp_smooth = np.swapaxes(interpXY, 0, 1)   
#             resp_pos = set_hyper2zero(resp_smooth)
#             
#             bf_level, resp_pos_level=[],[]
#             for level in range(301):
#                 """bf with center of mass"""
#                 freq_log = [math.log(i, 2) for i in x300]
#                 massX = freq_log*resp_pos[level]
#                 mass_sum = np.sum(resp_pos[level])
#                 mass_Xsum = np.sum(massX)
#                 center_mass = (2**(mass_Xsum/mass_sum))
#                 if self.check_authantic(center_mass, resp_pos[level]):
#                     center_mass = x300[np.argmax(resp_pos[level])]
#                 
#                 bf_level.append(center_mass)
#                 resp_pos_level.append(resp_pos[level])
#         
#         #bandwidth
#         def find_which_bin(bf, freq_bin):
#             for i,(f1,f2) in enumerate(zip(freq_bin[:-1], freq_bin[1:])):
#                 if f1 <= bf and f2 > bf:
#                     return i
#                     break
#                 elif (i==0 and f1>=bf) or (i==len(freq_bin[:-1])-1 and f2<=bf):
#                     return i
#                     break
#         
#         #minimum range with 67% weight
#         def find_range(arr, i, percentage=0.67):
#             total_sum = np.sum(arr)
#             target_sum = percentage * total_sum
#         
#             left, right = i, i
#             current_sum = arr[i]
#             #print(i, current_sum, target_sum)
#             while current_sum < target_sum:
#                 expand_left = left > 0
#                 expand_right = right < len(arr) - 1
#         
#                 # Decide which direction to expand
#                 if expand_left and expand_right:
#                     if arr[left - 1] > arr[right + 1]:  # Expand towards larger value
#                         left -= 1
#                         current_sum += arr[left]
#                     else:
#                         right += 1
#                         current_sum += arr[right]
#                 elif expand_left:
#                     left -= 1
#                     current_sum += arr[left]
#                 elif expand_right:
#                     right += 1
#                     current_sum += arr[right]
#                 else:
#                     break  # No more elements to expand
#                     
#             return left, right
#         
#         #minimum range above threshold
#         def find_range_around_peak(data, center_idx, threshold):
#             n = len(data)
#             left = center_idx
#             right = center_idx
#         
#             # Expand left
#             while left > 0 and data[left - 1] > threshold:
#                 left -= 1
#         
#             # Expand right
#             while right < n - 1 and data[right + 1] > threshold:
#                 right += 1
#         
#             return left, right
#         
#         
#         if not upscale:
#             left_level, right_level, band_level = [],[],[]
#             for level in range(6):
#                 x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
#                 bf, resp_pos = bf_level[level], resp_pos_level[level]
#                 threshold = np.max(resp_pos)/np.e
#                 
#                 point0 = find_which_bin(bf, x300)
#                 #left, right = find_range(resp_pos, point0, 1-1/np.e)
#                 left, right = find_range_around_peak(resp_pos, point0, threshold)
#                 band_level.append(math.log2(x300[right]/x300[left]))
#                 left_level.append(x300[left])
#                 right_level.append(x300[right])
#         
#         else:
#             left_level, right_level, band_level = [],[],[]
#             for level in range(301):
#                 x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
#                 bf, resp_pos = bf_level[level], resp_pos_level[level]
#                 threshold = np.max(resp_pos)/np.e
#                 
#                 point0 = find_which_bin(bf, x300)
#                 #left, right = find_range(resp_pos, point0, 1-1/np.e)
#                 left, right = find_range_around_peak(resp_pos, point0, threshold)
#                 band_level.append(math.log2(x300[right]/x300[left]))
#                 left_level.append(x300[left])
#                 right_level.append(x300[right])
#         
#         return bf_level, band_level, left_level, right_level
# =============================================================================
    
    
    #new, FFT upscale, return maximum and center mass, range bandwidth
    def get_level(self, on_off='on', ranges=(15,85), upscale=False):
        #bf
        if on_off=='on':
            working_resp = self.resp_on_mesh
        elif on_off=='off':
            working_resp = self.resp_off_mesh
        
        x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
        filt_resp = np.apply_along_axis(upscale_fft, 1, working_resp)
        filt_resp = pascal_smooth_loudness_large(filt_resp)
        #filt_resp = pascal_filter_convolve(filt_resp, weights=[1,4,6,4,1])
        
        if not upscale:
            step = 5
            resp_pos = set_hyper2zero(filt_resp)
            
        else:
            step = 301
            y300 = np.linspace(30,80,301)
            resp_300 = np.swapaxes(filt_resp, 0, 1)
            
            interpXY=[]
            for freq_layer in resp_300:
                interpXY.append(np.interp(y300, self.loud, freq_layer))
            interpXY = np.array(interpXY)
            
            interpXY = np.swapaxes(interpXY, 0, 1)   
            resp_pos = set_hyper2zero(interpXY)
        
        bf_level, center_mass_level, resp_pos_level=[],[],[]
        for level in range(step):
            """bf with maximum"""
            bf_max = x300[np.argmax(resp_pos[level])]
        
            """bf with center of mass"""
            freq_log = [math.log(i, 2) for i in x300]
            massX = freq_log*resp_pos[level]
            mass_sum = np.sum(resp_pos[level])
            mass_Xsum = np.sum(massX)
            center_mass = (2**(mass_Xsum/mass_sum))
        
            bf_level.append(bf_max)
            center_mass_level.append(center_mass)
            resp_pos_level.append(resp_pos[level])
        
        #bandwidth
        left_level, right_level, band_level = [],[],[]
        for level in range(step):
            cumsum = np.cumsum(resp_pos[level])
            total_depolar = cumsum[-1]
            left = np.searchsorted(cumsum, ranges[0]/100 * total_depolar)
            right = np.searchsorted(cumsum, ranges[1]/100 * total_depolar)
            
            bf_half = np.max(resp_pos[level])*0.5
            
            def expand_BW(bf_half, resp_pos, left, right):
                while (left>0 and resp_pos[left] > bf_half):
                    left -= 1
                while (right<len(resp_pos)-1 and resp_pos[right] > bf_half):
                    right += 1
                
                return left, right
            
            left, right = expand_BW(bf_half, resp_pos[level], left, right)
            
            
            band_level.append(math.log2(x300[right]/x300[left]))
            left_level.append(x300[left])
            right_level.append(x300[right])
        
        
        return bf_level, center_mass_level, band_level, left_level, right_level
    
    
# =============================================================================
#     #check if BW==0
#     def check_authantic(self, bf, filt1D, on_off='on', level=70):
#         x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
#         threshold = np.max(filt1D)/np.e
#         def find_which_bin(bf, freq_bin):
#             for i,(f1,f2) in enumerate(zip(freq_bin[:-1], freq_bin[1:])):
#                 if f1 <= bf and f2 > bf:
#                     return i
#                     break
#                 elif (i==0 and f1>=bf) or (i==len(freq_bin[:-1])-1 and f2<=bf):
#                     return i
#                     break
#         
#         def find_range_around_peak(data, center_idx, threshold):
#             n = len(data)
#             left = center_idx
#             right = center_idx
#         
#             # Expand left
#             while left > 0 and data[left - 1] > threshold:
#                 left -= 1
#         
#             # Expand right
#             while right < n - 1 and data[right + 1] > threshold:
#                 right += 1
#         
#             return left, right
# 
#         point0 = find_which_bin(bf, x300)
#         left, right = find_range_around_peak(filt1D, point0, threshold)
#         #if abs(left-x300[point0])<1 or abs(right-x300[point0])<1 or left==right:
#         if left==right:
#             
#             return 1
#         else:
#             return 0
# =============================================================================
        
# =============================================================================
#     #center of mass
#     def get_bf(self, on_off='on', level=70, output=False):
#         if on_off=='off':
#             working_resp = self.resp_off_mesh
#         else:
#             working_resp = self.resp_on_mesh
#         
#         x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
#         
#         Nzero = int(300/(len(self.freq)-1))-1
#         zero2D = np.zeros((6,len(self.freq),Nzero))        
#         upsampleX = np.dstack((working_resp, zero2D)).reshape((6,-1))[:,:301]
#         
#         level_idx = int((level/10)-3)
#         
#         filt1D = ndimage.gaussian_filter1d(upsampleX, Nzero)[level_idx]
#         resp_pos = set_hyper2zero(filt1D)
#         
#         """bf with center of mass"""
#         freq_log = [math.log(i, 2) for i in x300]
#         massX = freq_log*resp_pos
#         mass_sum = np.sum(resp_pos)
#         mass_Xsum = np.sum(massX)
#         center_mass = (2**(mass_Xsum/mass_sum))
#         
#         if self.check_authantic(center_mass, filt1D, on_off=on_off, level=level):
#             center_mass = x300[np.argmax(resp_pos)]
#         
#         if on_off=='on':
#             self.bf = center_mass
#         
#         if output:
#             return center_mass, resp_pos
# =============================================================================
        
# =============================================================================
#     #minimum range contain 67% weight
#     def get_bandwidth(self, on_off='on', output=False):        
#         """bandwidth using 67% total depolarization"""
#         """find start point on resp_on_mesh x axis"""
#         def find_which_bin(bf, freq_bin):
#             for i,(f1,f2) in enumerate(zip(freq_bin[:-1], freq_bin[1:])):
#                 if f1 < bf and f2 > bf:
#                     return i
#                     break
#         
#         def find_range(arr, i, percentage=0.67):
#             total_sum = np.sum(arr)
#             target_sum = percentage * total_sum
#         
#             left, right = i, i
#             current_sum = arr[i]
#             #print(i, current_sum, target_sum)
#             while current_sum < target_sum:
#                 expand_left = left > 0
#                 expand_right = right < len(arr) - 1
#         
#                 # Decide which direction to expand
#                 if expand_left and expand_right:
#                     if arr[left - 1] > arr[right + 1]:  # Expand towards larger value
#                         left -= 1
#                         current_sum += arr[left]
#                     else:
#                         right += 1
#                         current_sum += arr[right]
#                 elif expand_left:
#                     left -= 1
#                     current_sum += arr[left]
#                 elif expand_right:
#                     right += 1
#                     current_sum += arr[right]
#                 else:
#                     break  # No more elements to expand
#                     
#             return left, right
#         
#         x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
#         bf70, resp70_pos = self.get_bf(on_off, output=True)
#         point0 = find_which_bin(bf70, x300)
#         left, right = find_range(resp70_pos, point0, 1-1/np.e)
#         if on_off=='on':
#             self.bandwidth = math.log2(x300[right]/x300[left])
#             self.band_left = x300[left]
#             self.band_right = x300[right]
#         
#         if output:
#             return [x300[left], x300[right]]
# =============================================================================
        
# =============================================================================
#     #minimum range above threshold
#     def get_bandwidth(self, on_off='on', level=70, set_BW=True, output=False):
#         x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
#         
#         level_idx = int((level/10)-3)
#         
#         bf, resp_pos = self.get_bf(on_off, level=level, output=True)
#         
#         if on_off=='off':
#             working_resp = self.resp_off_mesh
#         else:
#             working_resp = self.resp_on_mesh
#         
#         
#         Nzero = int(300/(len(self.freq)-1))-1
#         zero2D = np.zeros((6,len(self.freq),Nzero))        
#         upsampleX = np.dstack((working_resp, zero2D)).reshape((6,-1))[:,:301]
#         filt1D = ndimage.gaussian_filter1d(upsampleX, Nzero)[level_idx]
#         
#         threshold = np.max(filt1D)/np.e
#         def find_which_bin(bf, freq_bin):
#             for i,(f1,f2) in enumerate(zip(freq_bin[:-1], freq_bin[1:])):
#                 if f1 <= bf and f2 > bf:
#                     return i
#                     break
#                 elif (i==0 and f1>=bf) or (i==len(freq_bin[:-1])-1 and f2<=bf):
#                     return i
#                     break
#         
#         def find_range_around_peak(data, center_idx, threshold):
#             n = len(data)
#             left = center_idx
#             right = center_idx
#         
#             # Expand left
#             while left > 0 and data[left - 1] > threshold:
#                 left -= 1
#         
#             # Expand right
#             while right < n - 1 and data[right + 1] > threshold:
#                 right += 1
#         
#             return left, right
# 
#         point0 = find_which_bin(bf, x300)
#         self.resp_temp = filt1D
#         left, right = find_range_around_peak(filt1D, point0, threshold)
#         
#         if set_BW:
#             self.bandwidth = math.log2(x300[right]/x300[left])
#             self.band_left = x300[left]
#             self.band_right = x300[right]
#         
#         if output:
#             return math.log2(x300[right]/x300[left]), x300[left], x300[right]
# =============================================================================
        
    def get_bf(self, on_off='on', level=70, set_bf=True, output=False):
        if on_off=='off':
            working_resp = self.resp_off_mesh
        else:
            working_resp = self.resp_on_mesh
        
        #x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
        x300 = np.logspace(math.log(3000,2), math.log(96000,2), 26, base=2)
        
        level60to80 = pascal_filter(working_resp[3:])
        #filt1D = upscale_fft(level60to80)
        filt1D = level60to80
        level_idx = int((level/10)-3)
        resp_pos = set_hyper2zero(filt1D)
        bf_max = x300[np.argmax(resp_pos)]
        
        """bf with center of mass"""
        freq_log = [math.log(i, 2) for i in x300]
        massX = freq_log*resp_pos
        mass_sum = np.sum(resp_pos)
        mass_Xsum = np.sum(massX)
        center_mass = (2**(mass_Xsum/mass_sum))
        
        if set_bf:
            self.bf = bf_max
        
        if output:
            return bf_max, center_mass, resp_pos, filt1D
    
    def get_bandwidth(self, on_off='on', level=70, ranges=(20,80), expand=True, set_BW=True, output=False):
        bf, center_mass, resp_pos, _ = self.get_bf(on_off, level=level, set_bf=False, output=True)
        #x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
        x300 = np.logspace(math.log(3000,2), math.log(96000,2), 26, base=2)
        cumsum = np.cumsum(resp_pos)
        total_depolar = cumsum[-1]
        left = np.searchsorted(cumsum, ranges[0]/100 * total_depolar)
        right = np.searchsorted(cumsum, ranges[1]/100 * total_depolar)
        
        if expand:
            bf_half = np.max(resp_pos)*0.5
            def expand_BW(bf_half, resp_pos, left, right):
                while (resp_pos[left] > bf_half and left>0):
                    left -= 1
                while (resp_pos[right] > bf_half and right<len(resp_pos)):
                    right += 1
                
                return left, right
            
            left, right = expand_BW(bf_half, resp_pos, left, right)
        
        if set_BW:
            self.bandwidth = math.log2(x300[right]/x300[left])
            self.band_left = x300[left]
            self.band_right = x300[right]
        
        if output:
            return math.log2(x300[right]/x300[left]), x300[left], x300[right]
    
    
    def plot_tuning_trace(self):
        bf, center_mass, resp_pos, resp_raw = self.get_bf(on_off='on', level=70, output=True)
        #_, x3, x4 = self.get_bandwidth(on_off='on', level=70, ranges=(20,80), expand=False, set_BW=False, output=True)
        
        
        #x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
        x300 = np.logspace(math.log(3000,2), math.log(96000,2), 26, base=2)
        
        xlabel = [3,6,12,24,48,96]
        xtick = [i * 1000 for i in xlabel]
        ytick = [30,40,50,60,70,80]
        
        #smooth = ndimage.gaussian_filter1d(resp_pos, sigma=2)
        cm_idx = find_which_bin(center_mass, x300)
        
        bf_idx = int(np.argmax(resp_pos))
        fig, ax = plt.subplots()
        ax.plot(x300, resp_raw)
        #ax.scatter(x300[cm_idx], resp_pos[int(cm_idx)], c='g', label='center mass')
        ax.scatter(x300[bf_idx], resp_pos[bf_idx], c='r', label='maximum')
        ax.hlines(y=-0.1, xmin=self.band_left, xmax=self.band_right, colors='orange', label='20-80%+expand')
        #ax.hlines(y=-0.2, xmin=x3, xmax=x4, colors='b', label='20-80%')
        
        ax.legend(loc='upper right')
        
        ax.set_xscale('log')
        ax.set_xticks(xtick)
        ax.set_xticklabels(xlabel)
        
        ax.set_title(f'{self.mouseID} - {self.filename}', fontsize=20)
        ax.set_xlabel('Frequency (kHz)', fontsize=18)
        ax.set_ylabel('Potential (mV)', fontsize=18)
        
        plt.savefig(f'{self.mouseID}_{self.filename}_tuning_trace.png', dpi=300, bbox_inches='tight')
        
        
# =============================================================================
#         
#         
#         
#         
#         
#         ax1.set_xscale('log')
#         ax1.minorticks_off()
#         ax1.set_xticks(xtick)
#         ax1.set_xticklabels(xlabel)
#         ax1.set_yticks(ytick)
#         ax1.set_title(f'{self.mouseID} - {self.filename}', fontsize=22)
#         ax1.set_xlabel('Frequency (kHz)', fontsize=20)
#         ax1.set_ylabel('Loudness (dB SPL)', fontsize=20)
#         
#         ax1.scatter(prop['center_mass'], prop['Y'], label='center_mass', s=8, c='forestgreen')
#         ax1.scatter(prop['bandwidth_left'], prop['Y'], label='left_edge', s=6, c='lawngreen')
#         ax1.scatter(prop['bandwidth_right'], prop['Y'], label='right_edge', s=6, c='lawngreen')
# 
#         
#         cax = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.03,ax1.get_position().height])
#         cbar = plt.colorbar(im, cax=cax)
#         cbar.ax.set_ylabel('mV')
# =============================================================================
        
        #plt.savefig(f'{self.mouseID}_{self.filename}_tuning.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    def get_tuning(self, on_off='on'):
        """
        Generate tuning curve on receptive field using center of mass and propotional deporlarization range.
        RF is upscaled from 6 sound level 26 frequency bin to 300 each for visual purpose.

        Parameters
        ----------
        show_bf : boolean, optional
            show bf and bandwidth using previous methods. The default is True.

        Returns
        -------
        

        """
        
        """upscale and smooth receptive field"""
        
        if on_off=='off':
            working_resp = self.resp_off_mesh
        else:
            working_resp = self.resp_on_mesh
        
        x300 = np.logspace(math.log(3000,2), math.log(96000,2), 301, base=2)
        y300 = np.linspace(30,80,301)
        bf_max, center_mass, _, bandwidth_left, bandwidth_right = self.get_level(upscale=True)
        
        filt1D = np.apply_along_axis(upscale_fft, 1, working_resp)
        filt1D = pascal_filter_level(filt1D)
        kernel = np.array([1, 4, 6, 4, 1])
        kernel = kernel / kernel.sum()
        bandwidth_left = pascal_filter_1D(bandwidth_left)
        bandwidth_right = pascal_filter_1D(bandwidth_right)
        
        resp_300 = np.swapaxes(filt1D, 0, 1)
        interpXY=[]
        for freq_layer in resp_300:
            interpXY.append(np.interp(y300, self.loud, freq_layer))
        interpXY = np.swapaxes(np.array(interpXY), 0, 1)
        resp_pos = set_hyper2zero(interpXY)
        
        contour_band = np.zeros((301,301))
        contour_bf = np.zeros((301,301))
        for y, (bf, bl, br) in enumerate(zip(bf_max, bandwidth_left, bandwidth_right)):
            x_bf = find_which_bin(bf, x300)
            contour_bf[y, x_bf] = 1 
            
            x_bl = find_which_bin(bl, x300)
            x_br = find_which_bin(br, x300)
            contour_band[y, x_bl] = 1
            contour_band[y, x_br] = 1
        
        self.resp_smooth = interpXY
        self.resp_pos = resp_pos
        
        
        self.tuning_property = {'X':x300, 'Y':y300, 'resp_upscale':interpXY, 
                                'bf_max':bf_max, 'center_mass':center_mass, 
                                'bandwidth_left':bandwidth_left, 'bandwidth_right':bandwidth_right,
                                'contour_bf':contour_bf, 'contour_bandwidth':contour_band}
        
        return self.tuning_property
    
    
    def plot_tuning(self, use_bf=True, on_off='on', saveplot=True):
        prop = self.get_tuning(on_off)
        
        XX, YY = np.meshgrid(prop['X'], prop['Y'])
        
        xlabel = [3,6,12,24,48,96]
        xtick = [i * 1000 for i in xlabel]
        ytick = [30,40,50,60,70,80]
    
        fig = plt.figure(figsize=(10,8))
        grid = plt.GridSpec(2, 1, hspace=0.6, height_ratios=[4,1])
        
        ax1 = fig.add_subplot(grid[0])
        #im = plt.pcolormesh(XX, YY, resp_smooth, cmap='RdBu_r', norm=colors.CenteredNorm())
        
        #ax1.add_collection(im)
        
        im = ax1.pcolormesh(XX, YY, prop['resp_upscale'], cmap='RdBu_r', norm=colors.CenteredNorm())
        ax1.set_xscale('log')
        ax1.minorticks_off()
        ax1.set_xticks(xtick)
        ax1.set_xticklabels(xlabel)
        ax1.set_yticks(ytick)
        #ax1.set_yticklabels(ylabel)
        ax1.set_title(f'{self.mouseID} - {self.filename}', fontsize=22)
        ax1.set_xlabel('Frequency (kHz)', fontsize=20)
        ax1.set_ylabel('Loudness (dB SPL)', fontsize=20)
        
# =============================================================================
#         ax1.scatter(prop['bf_max'], prop['Y'], label='bf_max', s=8, c='lightseagreen')
#         #ax1.scatter(prop['center_mass'], prop['Y'], label='center_mass', s=8, c='forestgreen')
#         ax1.scatter(prop['bandwidth_left'], prop['Y'], label='left_edge', s=6, c='lawngreen')
#         ax1.scatter(prop['bandwidth_right'], prop['Y'], label='right_edge', s=6, c='lawngreen')
# =============================================================================
# =============================================================================
#         masked_bf  = np.ma.masked_equal(prop['contour_bf'], 0)        
#         ax1.pcolormesh(XX, YY, masked_bf, c='Greys')
#         masked_bw  = np.ma.masked_equal(prop['contour_bandwidth'], 0)
#         ax1.pcolormesh(XX, YY, masked_bw, c='Greys')
# =============================================================================
        # Make a binary mask for contour plotting
        bf_mask = (prop['contour_bf'] != 0).astype(int)
        
        # Plot contour with thick turquoise outline
        ax1.contour(
            XX, YY, bf_mask,
            levels=[0.5],  # single contour line at 0.5 between 0 and 1
            colors=['turquoise'],
            linewidths=2.5  # adjust thickness here
        )
        bw_mask = (prop['contour_bandwidth'] != 0).astype(int)

        ax1.contour(
            XX, YY, bw_mask,
            levels=[0.5],
            colors=['limegreen'],
            linewidths=2.5
        )
        
        cax = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.03,ax1.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('mV')
        
        if saveplot:
            plt.savefig(f'{self.mouseID}_{self.filename}_tuning.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        
    def check_criterion(self):
        bf, _, resp_pos, _ = self.get_bf(on_off='on', level=70, set_bf=False, output=True)
        left, right = self.band_left, self.band_right
        
        XX = np.logspace(math.log(3000,2), math.log(96000,2), 26, base=2)
        left = find_which_bin(left, XX)
        right = find_which_bin(right, XX)
        
        depolar_inband = np.sum(resp_pos[left:right])/right-left
        depolar_exband = (np.sum(resp_pos[:left])+np.sum(resp_pos[right:]))/(left+len(resp_pos)-right)
        index = (depolar_inband - depolar_exband)/(depolar_inband + depolar_exband)
        
        return index

    def get_psth(self):
        self.get_resp_wwobfband()
        psth_in = np.mean(self.bfband['resp_in'], axis=0)
        psth_ex = np.mean(self.bfband['resp_ex'], axis=0)
        
        return [psth_in, psth_ex]
            
    

    def plot_psth(self, x_in_ms=True, saveplot=False):
        err = stats.sem(self.resp, axis=0)
        x = np.arange(len(self.psth))
        y = self.psth
        
        fig, ax = plt.subplots()
        ax.plot(x,y)
        ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
        #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.3) for _x in np.arange(0,5100,500)]
        [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [500,3000]]
        #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.3) for _x in np.arange(0,6000,50)]
        #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.7) for _x in np.arange(0,6000,250)]
        ax.set_title(f'{self.mouseID} - {self.filename} tone PSTH')   
        ax.set_xlim(0,10000)
        ax.set_ylabel('membrane potential (mV)')
        
        if x_in_ms:
            label = np.linspace(-20,380,11)
            ax.set_xticks(np.linspace(0,10000,11),label)
            ax.set_xlabel('time (ms)')
        else:
            ax.set_xticks([0,500,2000,3000,5000,7000,9000])
            ax.set_xlabel('data point (2500/100ms)')
            
        if saveplot:
            plt.savefig(f'{self.mouseID}_{self.filename}_tone-PSTH.svg', dpi=500, format='svg', bbox_inches='tight')
        
        plt.show()
        plt.clf()
        plt.close(fig)

    
    def plot_condition_psth(self, resp, x_in_ms=True, saveplot=False, condition='condition'):
        err = stats.sem(self.resp, axis=0)
        psth = np.mean(resp, axis=0)
        x = np.arange(len(psth))
        y = psth
        
        fig, ax = plt.subplots()
        ax.plot(x,y)
        ax.fill_between(x, y+err, y-err, color='orange', alpha=0.6)
        #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.3) for _x in np.arange(0,5100,500)]
        [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [500,3000]]
        #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.3) for _x in np.arange(0,6000,50)]
        #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.7) for _x in np.arange(0,6000,250)]
        ax.set_title(f'{self.mouseID} - {self.filename} tone PSTH -{condition}')   
        ax.set_xlim(0,10000)
        ax.set_ylabel('membrane potential (mV)')
        
        if x_in_ms:
            label = np.linspace(-20,380,11)
            ax.set_xticks(np.linspace(0,10000,11),label)
            ax.set_xlabel('time (ms)')
        else:
            ax.set_xticks([0,500,2000,3000,5000,7000,9000])
            ax.set_xlabel('data point (2500/100ms)')
            
        if saveplot:
            plt.savefig(f'{self.mouseID}_{self.filename}_tone-PSTH_{condition}.svg', dpi=500, format='svg', bbox_inches='tight')
        
        plt.show()
        plt.clf()
        plt.close(fig)

    def get_resp_wwobfband(self, use_band=True):
        if use_band:
            target_freq = [f for f in self.freq if f>self.band_left and f<self.band_right]
        else:
            for f1, f2 in zip(self.freq[:-1], self.freq[1:]):
                if f1 <= self.bf and f2 <= self.bf:
                    target_freq = [f1, f2]
        
        resp_in, resp_ex = [],[]
        para_in, para_ex = [],[]
        idx_in, idx_ex = [],[]
        
        for i,p in enumerate(self.para):
            if p[1] in target_freq:
                resp_in.append(self.resp[i])
                para_in.append(self.para[i])
                idx_in.append(i)
            else:
                resp_ex.append(self.resp[i])
                para_ex.append(self.para[i])
                idx_ex.append(i)
        
        self.bfband = {'idx_in':idx_in, 'resp_in':resp_in, 'para_in':para_in, 
                       'idx_ex':idx_ex, 'resp_ex':resp_ex, 'para_ex':para_ex}
        
    def plot_PSTH_wwobfband(self, x_in_ms=True, saveplot=False):
        self.get_resp_wwobfband()
        
        resp1 = self.bfband['resp_in']
        err1 = stats.sem(resp1, axis=0)
        psth1 = np.mean(resp1, axis=0)
        x1 = np.arange(len(psth1))
        y1 = psth1
        
        resp2 = self.bfband['resp_ex']
        err2 = stats.sem(resp2, axis=0)
        psth2 = np.mean(resp2, axis=0)
        x2 = np.arange(len(psth2))
        y2 = psth2
        
        #60-cycle filter
        from scipy.signal import iirnotch, filtfilt

        def filter_60hz(signal, fs, quality=30):
            freq = 60  # Hz
            b, a = iirnotch(freq, quality, fs)
            return filtfilt(b, a, signal)
        
        psth1 = filter_60hz(psth1, 25000, quality=2)
        
        fig, ax = plt.subplots()
        ax.plot(x1,y1,color='midnightblue')
        ax.fill_between(x1, y1+err1, y1-err1, color='cornflowerblue', alpha=0.6)
        ax.plot(x2,y2,color='firebrick')
        ax.fill_between(x2, y2+err2, y2-err2, color='salmon', alpha=0.6)
        #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.3) for _x in np.arange(0,5100,500)]
        [ax.axvline(x=_x, color='k', linestyle='--', alpha=0.5) for _x in [500,3000]]
        #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.3) for _x in np.arange(0,6000,50)]
        #[ax.axvline(x=_x, color='k', linestyle='--', alpha=0.7) for _x in np.arange(0,6000,250)]
        ax.set_title(f'{self.mouseID} - {self.filename} tone PSTH')   
        ax.set_xlim(0,10000)
        ax.set_ylabel('membrane potential (mV)')
        
        if x_in_ms:
            label = np.linspace(-20,380,11)
            ax.set_xticks(np.linspace(0,10000,11),label)
            ax.set_xlabel('time (ms)')
        else:
            ax.set_xticks([0,500,2000,3000,5000,7000,9000])
            ax.set_xlabel('data point (2500/100ms)')
            
        if saveplot:
            plt.savefig(f'{self.mouseID}_{self.filename}_tone_PSTH_wwoband.png', dpi=500, format='png', bbox_inches='tight')
        
        plt.show()
        plt.clf()
        plt.close(fig)
    
    def get_psth_cat(self, wwoband='in'):
        from IPython import get_ipython
        from matplotlib.lines import Line2D
        from matplotlib.widgets import Button, CheckButtons
        get_ipython().run_line_magic('matplotlib', 'qt5')

        if wwoband=='out':
            resp_work = self.bfband['resp_ex']
        else:
            resp_work = self.bfband['resp_in']
        y = np.mean(resp_work, axis=0)
        x = np.arange(len(y))
        
        # Initialize figure and axis
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        line, = ax.plot(x, y, label="Signal")
        
        # === CUSTOM CURSOR NAMES ===
        cursor_names = ["stim A", "stim B", "resp 1", "resp 2", "end"]
        NUM_CURSORS = len(cursor_names)
    
        cursor_lines = []
        cursor_labels = []
        cursor_positions = np.linspace(x[0], x[-1], NUM_CURSORS + 2)[1:-1]
        dragging = [None]
        active_cursor = [None]
        cursor_result = []
        done_flag = [False]
    
        ymin, ymax = ax.get_ylim()
        for i, xpos in enumerate(cursor_positions):
            # Draw vertical line
            line = Line2D([xpos, xpos], [ymin, ymax], color='r', linestyle='--', lw=1.5)
            ax.add_line(line)
            cursor_lines.append(line)
    
            # Label at middle y
            mid_y = (ymin + ymax) / 2
            label = ax.text(xpos, mid_y, cursor_names[i],
                            rotation=0, fontsize=10,
                            color='blue', ha='left', va='center',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            cursor_labels.append(label)
    
        # === Checkboxes as one horizontal row ===
        ax_check = plt.axes([0.05, 0.9, 0.9, 0.08])  # across the top
        checkbox_status = [False] * NUM_CURSORS
        check = CheckButtons(ax_check, cursor_names, checkbox_status)
        [lbl.set_fontsize(12) for lbl in check.labels]
    
        changing_checkbox = [False]
    
        def on_check(label):
            if changing_checkbox[0]:
                return
            idx = cursor_names.index(label)
            changing_checkbox[0] = True
            current_status = check.get_status()
            for i in range(NUM_CURSORS):
                if current_status[i] and i != idx:
                    check.set_active(i)  # toggle off
            # Re-read status after toggles
            new_status = check.get_status()
            active_cursor[0] = idx if new_status[idx] else None
            changing_checkbox[0] = False
    
        check.on_clicked(on_check)
    
        def on_press(event):
            if event.inaxes != ax or event.xdata is None:
                return
            if active_cursor[0] is not None:
                dragging[0] = active_cursor[0]
    
        def on_motion(event):
            if dragging[0] is None or event.inaxes != ax or event.xdata is None:
                return
            idx = dragging[0]
            x_new = event.xdata
            cursor_lines[idx].set_xdata([x_new, x_new])
            # Update label position
            cursor_labels[idx].set_position((x_new, (ymin + ymax) / 2))
            fig.canvas.draw_idle()
    
        def on_release(event):
            dragging[0] = None
    
        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("button_release_event", on_release)
    
        def output_and_close(event):
            for i, line in enumerate(cursor_lines):
                x_pos = line.get_xdata()[0]
                y_pos = np.interp(x_pos, x, y)
                cursor_result.append((cursor_names[i], x_pos, y_pos))
            print("Cursor values:")
            for name, xv, yv in cursor_result:
                print(f"{name}: x = {xv:.2f}, y = {yv:.2f}")
            done_flag[0] = True
            plt.close('all')
            get_ipython().run_line_magic('matplotlib', 'inline')
    
        ax_button = plt.axes([0.8, 0.05, 0.12, 0.075])
        button = Button(ax_button, "Output")
        button.on_clicked(output_and_close)
    
        plt.legend()
        plt.show(block=False)
    
        # Wait loop until output is clicked
        while not done_flag[0] and plt.fignum_exists(fig.number):
            plt.pause(0.1)
    
        return cursor_result


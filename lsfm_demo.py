from TDMS import Tdms
import numpy as np
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy import stats
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import TFTool
import pandas as pd
import lsfm_analysis
from scipy.signal.windows import dpss



if  __name__ == "__main__":
    df = pd.read_csv('patch_list_USBMAC.csv', dtype={'date':str, '#':str})
    ilsfm = df.index[df['type']=='Log sFM']
    fdir = df['path'][45]
    t = Tdms()
    t.loadtdms(fdir, load_sound=False)
    _,para = t.get_stim()
    resp,_ = t.get_dpk()
    fs = 25000
    resp_z = stats.zscore(resp)
    resp_z = signal.resample(resp_z, 1200, axis=1)
    resp_pad = np.pad(resp_z, [(0,0), (0,len(resp_z[0]))], 'constant')
    resp_fft = np.abs(np.fft.fft(resp_pad)**2)
    freq = np.fft.fftfreq(len(resp_pad[0]), d=1/600)
    mask = freq>=0
    
    """Merge 1
    res, prop = lsfm_analysis.para_merge(para, resp_fft, axis=0)
    power = []
    target_freq = np.arange(1.0,257.0)
    ext_freq = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    label_freq = [0.0, 1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
    oct_freq = [4.0, 32.0, 256.0]
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
            plt.axvline(x=xc, color='k', linestyle='--', alpha=0.3)
        for xc in oct_freq:
            plt.axvline(x=xc, color='r', linestyle='--', alpha=0.3)
        plt.show()
       """
    
    """merge 2"""
    res, prop = lsfm_analysis.para_merge2(para, resp_fft, axis=2)
    power = []
    target_freq = np.arange(1.0,257.0,0.5)
    oct_freq = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    label_freq = [0.0, 1.0, 2.0, 8.0, 16.0, 64.0, 128.0]
    multi_freq = [4.0, 32.0, 256.0]
    
    idx_freq = [i for i, a in enumerate(freq) if a in oct_freq]
    idx_freq = np.array(idx_freq)
    pow_diff = []
    
    for pp in res:
        #diff = (3*pp[idx_freq] - pp[idx_freq-1] - pp[idx_freq-2] - pp[idx_freq+2])/pp[idx_freq]
        diff = pp[idx_freq]/pp[idx_freq-2] + pp[idx_freq]/pp[idx_freq+2]
        for i, a in enumerate(diff):
            if a >= 0:
                diff[i] = np.log(diff[i])
            elif a < 0:
                diff[i] = -1*np.log(-1*diff[i])
        pow_diff.append(diff)
    pow_diff = np.array(pow_diff)
        
    mod_rate = sorted(list(set(prop['set1'])))
    
    for j in np.arange(1,7):
        x,y=[],[]
        idx = [i for i, a in enumerate(prop['set1']) if a == mod_rate[j]]
        for i in idx:
            y.append([prop['set2'][i]]*9)
            x.append(pow_diff[i,:])
        for k in range(13):
            plt.scatter(oct_freq, y[k], c=100*x[k], s=10*np.abs(x[k]), cmap='bwr')      
        ax = plt.subplot()
        txt = (f'Modulation: {mod_rate[j]} Hz')
        ax.text(80,0.45, txt, horizontalalignment='left')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Resp Freq (Hz)')
        plt.ylabel('Center Freq (KHz)')
        plt.colorbar()
        for xc in multi_freq:
            plt.axvline(x=xc, color='r', linestyle='--', alpha=0.5)
        plt.show()

            
# =============================================================================
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.scatter3D(prop['set1'], prop['set2'],1.0, c=dif[:,0])
#     ax.scatter3D(prop['set1'], prop['set2'],2.0, c=dif[:,1])
#     ax.scatter3D(prop['set1'], prop['set2'],4.0, c=dif[:,2])
#     ax.scatter3D(prop['set1'], prop['set2'],8.0, c=dif[:,3])
#     ax.scatter3D(prop['set1'], prop['set2'],16.0, c=dif[:,4])
#     ax.scatter3D(prop['set1'], prop['set2'],32.0, c=dif[:,5])
#     ax.scatter3D(prop['set1'], prop['set2'],64.0, c=dif[:,6])
#     ax.scatter3D(prop['set1'], prop['set2'],128.0, c=dif[:,7])
#     ax.scatter3D(prop['set1'], prop['set2'],256.0, c=dif[:,8])
#     ax.set_xlabel('modulation rate')
#     ax.set_ylabel('center frequency')
#     ax.set_zlabel('fft frequency')
# =============================================================================


# =============================================================================
#     idx_freq = [i for i,a in enumerate(freq) if a in target_freq]
# 
#     power_at_freq=[]
#     for pp in res:
#         power_at_freq.append(pp[idx_freq])
# 
#     for i,a in enumerate(power_at_freq):
#         plt.scatter(target_freq, a, s=12)
#         plt.xscale('log')
#         plt.yscale('log')
#         ax = plt.subplot()
#         if prop['axis'] == 0:
#             txt = prop['parameter'] + '\n %.1f kHz' % prop['set1'][i] + '\n %.5f octave' % prop['set2'][i]
#         elif prop['axis'] == 1:
#             txt = prop['parameter'] + '\n %i Hz' % prop['set1'][i] + '\n %.5f octave' % prop['set2'][i]
#         else:
#             txt = prop['parameter'] + '\n %i Hz' % prop['set1'][i] + '\n %.1f kHz' % prop['set2'][i]
#         ax.text(0.95,0.85, txt, transform=ax.transAxes, horizontalalignment='right')
#         for xc in label_freq:
#             plt.axvline(x=xc, color='k', linestyle='--', alpha=0.3)
#         for xc in oct_freq:
#             plt.axvline(x=xc, color='r', linestyle='--', alpha=0.3)
#         plt.show()
# =============================================================================
        
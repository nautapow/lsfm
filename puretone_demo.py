from TDMS import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import scipy.io
import TFTool


def mem_V(stim, para, resp):
    on_r, off_r = [],[]
    sum_on, sum_off = [],[]
    on_p, on_m, off_p, off_m = [],[],[],[]
    
    #use the sum of PSP amplitude to plot character frequency
    #use 20ms at begining to get baseline for substraction
    #on = from onset of sound stimulus to 94ms later
    #off = offset of sound with 100ms duration
    for i in range(len(resp)):
        base = np.mean(resp[i][:500])
        on_r.append(resp[i][800:3150]-base)
        off_r.append(resp[i][3150:5650]-base)
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
            
        
        
    file_name = file_dir[file_dir.find('2021', -40):file_dir.find('2021', -40)+12]
    
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
    ax.text(0.05, 0.018, file_name, fontsize=10, transform=ax.transAxes)
    ax.text(0.2,0.018, 'on', fontsize=10, transform=ax.transAxes)
    plt.show()
    plt.clf()
    
    
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
    ax.text(0.05, 0.018, file_name, fontsize=10, transform=ax.transAxes)
    ax.text(0.2,0.018, 'off', fontsize=10, transform=ax.transAxes)
    plt.show()
    plt.clf()
    

def avg_freq(stimulus, parameter, response):
    _r = response[:]
    del _r[::71]
    _r = np.array(_r)
    resp_avg = np.mean(_r.reshape(-1, 7, 10000), axis = 1)
    
    _p = parameter[:]
    del _p[::71]
    _p = np.array(_p)
    para_avg = np.mean(_p.reshape(-1,7,3), axis = 1, dtype=np.int32)
    
    _s = stimulus[:]
    del _s[::71]
    _s = np.array(_s)
    stim_avg = np.mean(_s.reshape(-1, 7, 10000), axis = 1)
    
    return stim_avg, para_avg, resp_avg


'''        
    #for counting the total number of dB and frequency used
    loud, freq, _ = zip(*para)
    n_loud = [[x, loud.count(x)] for x in set(loud)]
    n_loud.sort()
    _n = [x[1] for x in n_loud]
    _n = max(_n)
'''



if  __name__ == "__main__":
    file_dir = r'/Users/POW/Desktop/python_learning/0622_puretone.tdms'
    #file_dir =r'E:\Documents\PythonCoding\puretone_0622.tdms'
    #file_dir =r'Q:\[Project] 2020 in-vivo patch with behavior animal\Raw Results\20210812\20210812_004_2021_08_12_12_58_23.tdms'
    #file_dir='/Volumes/BASASLO/in-vivo_patch_result/20211018/20211018_001_2021_10_18_11_52_22.tdms'
    t1= Tdms(file_dir)
    t1.loadtdms(protocol=2)
    
    stim, para = t1.get_stim()
    resp = t1.get_resp()
    sound = t1.get_sound()
    mem_V(stim, para, resp)
    mem_V(*avg_freq(stim, para, resp))
    _,_para_avg,_resp_avg = avg_freq(stim, para, resp)
    
    #plot average resoonse of same frequency
    fig = plt.figure()
    ax1 = plt.subplot()
    legend = str(range(30,100,10))
    for i in range(10):
        for j in range(i,70,10):
            plt.plot(_resp_avg[j]-_resp_avg[j][0], label = '%s'
                     %_para_avg[j][0])
            
        ax1.legend() 
        ax2 = plt.subplot()
        ax2.text(0.9,1,_para_avg[j][1], transform=ax2.transAxes,)
        plt.savefig('tone_resp_%s.png' %i, dpi=300)
        plt.clf()
    
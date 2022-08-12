import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import scipy.io
import TFTool
import pandas as pd
import math

def strf(resp, cwt, filename, plot=True, saveplot=False):
        f = cwt['f']
        wt = cwt['wt']
        f = f[::-1]
        wt = [i[::-1] for i in wt]
        
        t_for,t_lag = 0.1, 0.4
        fs = 25000
        
        corr_allstim=[]
        for i, single_stim in enumerate(wt):
            stim_pad = np.pad(single_stim, pad_width=[(0,0), (int(t_lag*fs),int(t_for*fs))], mode='constant', constant_values=0)
            
            corr_siglestim=[]
            for freq_band in stim_pad:
                corr = np.correlate(freq_band, resp[i], mode='valid')
                corr_siglestim.append(corr)
            
            corr_allstim.append(corr_siglestim)
        
        coeff = np.array(corr_allstim)
        coeff = np.mean(coeff, axis=0)
            
        plt.imshow(coeff, aspect='auto', origin='lower')
        ylabel = [round(i/1000, 2) for i in f[::20]]
        plt.yticks(np.linspace(0,len(f), len(ylabel)), ylabel)
        xlabel = np.linspace(-0.1,0.4,6)
        xlabel = [round(i,2) for i in xlabel]
        plt.xticks(np.linspace(0,12500,6), xlabel)
        plt.colorbar()
        
        if saveplot:
            plt.savefig(f'{filename}_strf.png', dpi=500, bbox_inches='tight')
            if plot:
                plt.show()
            plt.clf()
        
        elif plot:
            plt.show()
        else:
            pass
        
        return coeff
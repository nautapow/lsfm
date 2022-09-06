import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal
from scipy import stats
import scipy.io
import TFTool
import pandas as pd
import math

class STRF():
    def __init__(self, cwt, resp, filename, forward=0.1, delay=0.4, fs=25000):
        f = cwt['f']
        wt = cwt['wt']
        f = f[::-1]
        wt = [i[::-1] for i in wt]
        self.fs = fs
        self.point = int((forward+delay)*fs)
        self.stim = wt
        self.resp = resp
        self.freq = f
        self.filename = filename
        self.t_for = forward
        self.t_lag = delay
        

    def strf(self, plot=True, saveplot=False):        
            resp_zscore = stats.zscore(self.resp, axis=0)
            corr_allstim=[]
            for i, single_stim in enumerate(self.stim):
                stim_pad = np.pad(single_stim, pad_width=[(0,0), (int(self.t_lag*self.fs), int(self.t_for*self.fs))], mode='constant', constant_values=0)
                
                corr_siglestim=[]
                for freq_band in stim_pad:
                    corr = signal.correlate(freq_band, self.resp[i], mode='valid', method='fft')
                    corr_siglestim.append(corr)
                
                corr_allstim.append(corr_siglestim)
            
            coeff = np.array(corr_allstim)
            coeff = np.flip(np.mean(coeff, axis=0), axis=1)
                
            plt.imshow(coeff, aspect='auto', origin='lower')
            ylabel = [round(i/1000, 2) for i in self.freq[::20]]
            plt.yticks(np.linspace(0,len(self.freq), len(ylabel)), ylabel)
            xlabel = np.linspace(-1*self.t_for, self.t_lag, 6)
            xlabel = [round(i,2) for i in xlabel]
            plt.xticks(np.linspace(0,self.point,6), xlabel)
            plt.colorbar()
            
            if saveplot:
                plt.savefig(f'{self.filename}_strf.png', dpi=500, bbox_inches='tight')
                if plot:
                    plt.show()
                plt.clf()
            
            elif plot:
                plt.show()
            else:
                pass
            
            return coeff
        
    
    def fake_strf(self, plot=False, saveplot=False):
        """artificial STRF"""
        
        delay = np.linspace(-1*self.t_for, self.t_lag, int(self.point+1))
        grid = np.array(np.meshgrid(delay, self.freq))
        Xv, Yv = np.meshgrid(delay, self.freq)
        grid = grid.swapaxes(0, -1).swapaxes(0, 1)
        rf_para = (0.15, 6000, .001, 1000)
        
        means_excit = [rf_para[0], rf_para[1]]
        means_inhib1 = [0.15, 5000]
        means_inhib2 = [0.1, 7000]
        cov = [[rf_para[2],0], [0, rf_para[3]]]
        cov2 = [[.004,0], [0, 100000]]
        
        gauss_excit = stats.multivariate_normal.pdf(grid, means_excit, cov)
        gauss_inhib1 = -1*stats.multivariate_normal.pdf(grid, means_inhib1, cov2)
        gauss_inhib2 = -1*stats.multivariate_normal.pdf(grid, means_inhib2, cov2)
        
        noise = np.random.randn(len(self.freq), len(delay))/100
        
        #strf = gauss_inhib1 + gauss_excit + gauss_inhib2
        strf = gauss_excit #+ noise
        strf = np.flip(strf, axis=1)
        

        plt.imshow(strf, aspect='auto', origin='lower', norm=colors.CenteredNorm())
        ylabel = [round(i/1000, 2) for i in self.freq[::20]]
        plt.yticks(np.linspace(0,len(self.freq), len(ylabel)), ylabel)
        xlabel = np.linspace(-0.1,0.4,6)
        xlabel = [round(i,2) for i in xlabel]
        plt.xticks(np.linspace(0,self.point+1,6), xlabel)
        plt.colorbar()
        
        if saveplot:
            plt.savefig(f'{self.filename}_fake_strf.png', dpi=500, bbox_inches='tight')
            if plot:
                plt.show()
            plt.clf()
        elif plot:
            plt.show()
            plt.clf()
        
        return strf, rf_para
    
    
    def resp_simu(self, strf):
        """Resp simulated by STRF"""
        self.strf = strf
        resp_simus=[]
        # delay = np.linspace(-1*self.t_for, self.t_lag, self.point+1)
        
        for i,stim_wt in enumerate(self.stim):
         
            npad = ((0,0), (int(0.4*self.fs), int(0.1*self.fs)))
            stim_pad = np.pad(stim_wt, pad_width=npad, mode='constant', constant_values=0)
            
            freq_band_conv=[]
            for j, f_band in enumerate(stim_pad):
                fbc = signal.convolve(f_band, strf[j], mode='valid', method='fft')
                freq_band_conv.append(fbc)
            
            freq_band_conv = np.array(freq_band_conv)
            simulated = np.sum(freq_band_conv, axis=0)
            resp_simus.append(simulated)
            
        return resp_simus
    
    
    
    
    
    
    
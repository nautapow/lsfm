from TDMS import Tdms
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import scipy.io
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
import pandas as pd

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
    
    properties = {'axis': s, 'set': value_set}
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
        s = 'Center Frequency+Bandwidth'
    elif axis == 1:
        value_set1 = sorted(set(mod))
        value_set2 = sorted(set(bd))
        obj1 = 2
        obj2 = 1
        s = 'Modulation Rate+Bandwidth'
    elif axis == 2:
        value_set1 = sorted(set(mod))
        value_set2 = sorted(set(cf))
        obj1 = 2
        obj2 = 0
        s = 'Modulation Rate+Center Frequency'
    else:
        raise KeyError('Please enter the correct axis code')
    
    res=[]
    for idx, par in enumerate(para):
        if par[obj1] in value_set1 and par[obj2] in value_set2:
            res.append(resp[idx])
        
        mean_resp = np.mean(res, axis=0)
    
    properties = {'axis': s, 'set1': value_set1, 'set2' :value_set2}
    return mean_resp, properties


if  __name__ == "__main__":
    #os.chdir('q:\[Project] 2020 in-vivo patch with behavior animal\Raw Results')
    #path = os.getcwd()
    
    #t1 = Tdms('/Users/POW/Documents/Python_Project/lsfm_analysis/20210730_002_2021_07_30_13_53_09.tdms')
    #t1 = Tdms('/home/cwchiang/repos/lsfm/lsfm.tdms)
    #t1 = Tdms(r'E:\Documents\PythonCoding\20210730_002_2021_07_30_13_53_09.tdms')
    df = pd.read_csv('patch_list_USBMAC.csv', dtype = {'date':str, '#':str})
    index = df.index[df['type']=='Log sFM']
    for i in index:
        try:
            path = df['path'][i]
            t = Tdms()
            t.loadtdms(path, load_sound=False)
            stim, para = t.get_stim()
            resp,_ = t.get_dpk()
            
            mean, prop = para_merge(para, resp, axis=0)
            for idx, res in enumerate(mean):
                plt.plot(res)
                ax = plt.subplot()
                txt = df['date'][i]+'_'+df['#'][i] + '\n ' + \
                    prop['axis']+' '+str(prop['set'][idx])
                ax.text(0.05,0.9,txt,transform=ax.transAxes, fontsize=10)
                plt.show()
                plt.clf()
            
            mean, prop = para_merge(para, resp, axis=1)
            for idx, res in enumerate(mean):
                plt.plot(res)
                ax = plt.subplot()
                txt = df['date'][i]+'_'+df['#'][i] + '\n ' + \
                    prop['axis']+' '+str(prop['set'][idx])
                ax.text(0.05,0.9,txt,transform=ax.transAxes, fontsize=10)
                plt.show()
                plt.clf()
            
            mean, prop = para_merge(para, resp, axis=2)
            for idx, res in enumerate(mean):
                plt.plot(res)
                ax = plt.subplot()
                txt = df['date'][i]+'_'+df['#'][i] + '\n ' + \
                    prop['axis']+' '+str(prop['set'][idx])
                ax.text(0.05,0.9,txt,transform=ax.transAxes, fontsize=10)
                plt.show()
                plt.clf()
                
        except:
            pass
    




# =============================================================================
# cwt = scipy.io.loadmat('/Users/POW/Desktop/python_learning/cwt_sound.mat')
# f = cwt['f']
# 
# n_epochs = len(resp)
# wt = []
# R = []
# for x in range(n_epochs):
#     R.append(resp[x])
#     wt.append(cwt['wt'][0][:][x][:])
#     
# #cwt_f = np.array(y.T)[:][0]
# 
# R = np.array(R)
# wt = np.array(wt)
# R = signal.resample(R, 500, axis=1)
# P = wt**2
# 
# tmin = -0.1
# tmax = 0.4
# sfreq = 250
# freqs = f.T[:][0]
# 
# train, test = np.arange(n_epochs - 1), n_epochs - 1
# X_train, X_test, y_train, y_test = P[train], P[test], R[train], R[test]
# X_train, X_test, y_train, y_test = [np.rollaxis(ii, -1, 0) for ii in
#                                     (X_train, X_test, y_train, y_test)]
# # Model the simulated data as a function of the spectrogram input
# alphas = np.logspace(-3, 3, 7)
# scores = np.zeros_like(alphas)
# models = []
# for ii, alpha in enumerate(alphas):
#     rf = ReceptiveField(tmin, tmax, sfreq, freqs, estimator=alpha)
#     rf.fit(X_train, y_train)
# 
#     # Now make predictions about the model output, given input stimuli.
#     scores[ii] = rf.score(X_test, y_test)
#     models.append(rf)
# 
# times = rf.delays_ / float(rf.sfreq)
# 
# # Choose the model that performed best on the held out data
# ix_best_alpha = np.argmax(scores)
# best_mod = models[ix_best_alpha]
# coefs = best_mod.coef_[0]
# best_pred = best_mod.predict(X_test)[:, 0]
# 
# # Plot the original STRF, and the one that we recovered with modeling.
# 
# plt.pcolormesh(times, rf.feature_names, coefs, shading='auto')
# #plt.set_title('Best Reconstructed STRF')
# #plt.autoscale(tight=True)
# strf_o = {'time' : times, 'feature' : rf.feature_names, 
#           'coef' : coefs}
# plt.yscale('log')
# plt.ylim(2000,90000)
# plt.savefig('strf.png', dpi=300)
# scipy.io.savemat('strf_out.mat', strf_o)
# 
# 
# '''
# #for corrilation
# t2 = Tdms(path)
# t2.loadtdms()
# 
# R = []
# for x in np.arange(len(t1.get_res())):
#     try:
#         r = np.corrcoef(t1.get_res()[x], t2.get_res()[x])
#     except ValueError:
#         print(x)
#     else:
#         R.append(r[0,1])
#         
# folder1 = os.path.dirname(t1.get_path())[-8:]
# folder2 = os.path.dirname(t2.get_path())[-8:]
# 
# for x in np.arange(len(R)):
#     plt.scatter(x, R[x], c='black', s=5)
# 
# ax = plt.subplot()
# txtstr = folder1 + ' vs ' + folder2
# ax.text(0.98, 0.06, txtstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', horizontalalignment='right', color='red')
# plt.savefig(txtstr, dpi=300)
# '''
# =============================================================================

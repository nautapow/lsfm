import nems
from nems.models.base import Model
from nems.layers import FiniteImpulseResponse, DoubleExponential, WeightChannels
from nems.models import LN_STRF
from nems.tools.demo_data.file_management import download_demo, load_demo
from nems.metrics import correlation
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
import sys


sys.path.append(r'C:\Users\McGinley3\Documents\GitHub\lsfm')
import TFTool

df = pd.read_csv('patch_list_E.csv', dtype={'date':str, '#':str})
df_loc = 74
filename = df['filename'][df_loc]
version = df['Version'][df_loc]
cell_data = np.load(f'{filename}_lsfm.npy', allow_pickle=True)

para = cell_data.item().get('para')
stim = cell_data.item().get('stim')
resp = cell_data.item().get('resp')
stim_fir = cell_data.item().get('stim_fir')
resp = TFTool.prefilter(resp, 25000)

import mat73
data = scipy.io.loadmat('20220527_avg_data.mat')
cwt = mat73.loadmat('20220527_avg_cwt.mat')
#cwt = mat73.loadmat(f'{filename}_cwt.mat')
stim_wt = np.array(cwt['wt'])
resp = np.array(data['resp'])

model = Model()
model.add_layers(
    #WeightChannels(shape=(121,30)),  # 18 spectral channels
    FiniteImpulseResponse(shape=(30, 121)),  # 15 taps
    DoubleExponential(shape=(1,))           # static nonlinearity, 1 output
)

model=model.sample_from_priors()
fir_all = []
wc_all = []
strf_all = []

for i in range(len(resp)):
    ss = signal.resample(stim_wt[i], 5000, axis=1)
    rr = signal.resample(resp[i], 5000)
    stim4fit = np.swapaxes(ss, 0, 1)
    resp4fit = rr
    
    # temp fix -- stretch trials into a single dimension
    # correct method should use batches
    
    fitted_model = model.copy()
    fitted_model = fitted_model.fit(stim4fit, resp4fit, backend='scipy',
                             fitter_options={'options':{'maxiter': 500, 'maxfun': 2000,
                                                        'gtol': 0.00001, 'ftol': 0.00001}})
    
    #wc=fitted_model.get_parameter_values()['WeightChannels']['coefficients']
    fir=fitted_model.get_parameter_values()['FiniteImpulseResponse']['coefficients']
    fir_all.append(fir)
    #wc_all.append(wc)  
    #strf = wc @ fir.T
    #strf_all.append(strf)
    
    print(i)

strf = np.nanmean(fir_all, axis=0)
xx, yy = np.meshgrid(range(30), range(121))
plt.pcolormesh(xx, yy, strf.T)
plt.savefig("strf_20220527_avg_5000datapoint.png", dpi=500, bbox_inches='tight')
np.save('fir_all_20200527_avg', fir_all, allow_pickle=True)
#plt.imshow(strf,aspect='auto',origin='lower')

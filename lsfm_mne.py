import numpy as np
import scipy.io
from scipy import signal
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge
import matplotlib.pyplot as plt



cwt = scipy.io.loadmat('/Users/POW/Desktop/python_learning/cwt_sound.mat')
f = cwt['f']

n_epochs = len(resp)
wt = []
R = []
for x in range(n_epochs):
    R.append(resp[x])
    wt.append(cwt['wt'][0][:][x][:])
    
#cwt_f = np.array(y.T)[:][0]

R = np.array(R)
wt = np.array(wt)
R = signal.resample(R, 500, axis=1)
P = wt**2

tmin = -0.1
tmax = 0.4
sfreq = 250
freqs = f.T[:][0]

train, test = np.arange(n_epochs - 1), n_epochs - 1
X_train, X_test, y_train, y_test = P[train], P[test], R[train], R[test]
X_train, X_test, y_train, y_test = [np.rollaxis(ii, -1, 0) for ii in
                                    (X_train, X_test, y_train, y_test)]
# Model the simulated data as a function of the spectrogram input
alphas = np.logspace(-3, 3, 7)
scores = np.zeros_like(alphas)
models = []
for ii, alpha in enumerate(alphas):
    rf = ReceptiveField(tmin, tmax, sfreq, freqs, estimator=alpha)
    rf.fit(X_train, y_train)

    # Now make predictions about the model output, given input stimuli.
    scores[ii] = rf.score(X_test, y_test)
    models.append(rf)

times = rf.delays_ / float(rf.sfreq)

# Choose the model that performed best on the held out data
ix_best_alpha = np.argmax(scores)
best_mod = models[ix_best_alpha]
coefs = best_mod.coef_[0]
best_pred = best_mod.predict(X_test)[:, 0]

# Plot the original STRF, and the one that we recovered with modeling.

plt.pcolormesh(times, rf.feature_names, coefs, shading='auto')
#plt.set_title('Best Reconstructed STRF')
#plt.autoscale(tight=True)
strf_o = {'time' : times, 'feature' : rf.feature_names, 
          'coef' : coefs}
plt.yscale('log')
plt.ylim(2000,90000)
plt.savefig('strf.png', dpi=300)
scipy.io.savemat('strf_out.mat', strf_o)
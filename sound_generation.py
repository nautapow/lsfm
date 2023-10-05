import numpy as np
from scipy.integrate import quad
import sympy as smp
import math
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io
import pyaudio
from numba import jit
import time

@jit
def freq_mod_wave(time, fc, bw, fm):
    #t = np.arange(time,1)
    f_t = fc*bw**(0.5*(np.sin(2*math.pi*fm*time)))
    
    return f_t


def lsfm_waveform(t, fs, fc, bw, fm):
    x_t=[]
    for x in np.arange(t, step=1/fs):
        phi = quad(freq_mod_wave, 0,x, args=(fc, bw, fm), limit=100)[0]
        x_t.append(np.sin(2*math.pi*phi))
    
    return np.array(x_t)

# =============================================================================
# t = smp.symbols('t', real=True)
# fc = smp.symbols('fc', real=True)
# bw = smp.symbols('bw', real=True)
# fm = smp.symbols('fm', real=True)
# ft = fc*bw**((1/2)*smp.sin(2*smp.pi*fm*t))
# smp.integrate(ft,t)
# =============================================================================

fs=200000

test1 = lsfm_waveform(1, fs, 48000, 2, 4)
test1 = lsfm_waveform(1, fs, 1000, 1, 4)
test2 = lsfm_waveform(1, fs, 1000, 3, 4)
test3 = lsfm_waveform(1, fs, 1000, 5, 4)

mdic = {'sound3':test2}
scipy.io.savemat('test_sound.mat', mdic)

samples1 = (0.02*test1).astype(np.float32).tobytes()
samples2 = (0.02*test2).astype(np.float32).tobytes()
samples3 = (0.05*test3).astype(np.float32).tobytes()
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

stream.write(samples1)
stream.write(samples2)
stream.write(samples3)

stream.stop_stream()
stream.close()

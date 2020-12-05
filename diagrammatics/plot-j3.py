
#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy
from scipy.fftpack import fftfreq, fftshift
import os
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
import matplotlib as mpl

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
tableau10 = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f']#, '#bab0ac']
tableauCB = ['#1170aa', '#fc7d0b', '#a3acb9', '#57606c', '#5fa2ce', '#c85200', '#7b848f', '#a3cce9', '#ffbc79', '#c8d0d9']
mpl.rcParams['axes.prop_cycle'] = cycler(color=tableau10)

SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIGGER_SIZE = 15

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

folder = 'j3_g10_5_v0p2'
folder = 'j3_g10_5_v0'
files = glob.glob(f'{folder}/*.pickle')

save_plots = False

plt.ion()

def savefig(fname, transparent=True):
    if save_plots:
        plt.savefig(fname, transparent=transparent)

def nmax(x):
    return np.max(np.abs(x))

def nm(x):
    return x / np.max(np.abs(x))

def fft(t,f, inverse=False):
    T = max(t) - min(t)
    Nt = len(t)
    dt = T/Nt

    xw = np.arange(Nt) * 2 * np.pi / T


    if inverse:
        xw = np.concatenate([xw[:Nt//2]+2*np.pi/dt, xw[Nt//2:]])
        idx = np.argsort(xw)
        xw = xw[idx]
        f = f[idx]
        fw = scipy.fft.ifft(f, axis=0)/np.sqrt(Nt)*Nt
    else:
        fw = scipy.fft.fft(f, axis=0)/np.sqrt(Nt)
        xw = np.concatenate([xw[:Nt//2], xw[Nt//2:]-2*np.pi/dt])
        idx = np.argsort(xw)
        xw = xw[idx]
        fw = fw[idx]
    return xw, fw

def next(name):
    i = 0
    while f'{name}_{i}.pdf' in os.listdir(f'{folder}/figs/'):
        i += 1
    return f'{folder}/figs/{name}_{i}.pdf'

res = []
for f in files:
    reader = open(f,'rb')
    try:
        while True:
            a = pickle.load(reader)
            res.append(a)

    except:
        reader.close()

w = []
T = []
jh = []
jl = []
jl2 = []
jqp = []
durations = []

for r in res:
    w.append(r['w'])
    T.append(r['T'])

    jh.append(r['jH'])
    jl.append(r['jL'])
    jl2.append(r['jL2'])
    jqp.append(r['jQP'])

    durations.append(r['duration'])

w = np.array(w)
T = np.array(T)
jl = np.array(jl)
jl2 = np.array(jl2)
jh = np.array(jh)
jqp = np.stack(jqp)

if len(np.unique(w))>1:
    x = np.copy(w)
    xlabel = '$\omega$'
    omega = True
else:
    x = np.copy(T)
    xlabel = '$T$'
    omega = False

ind = np.argsort(x)

x = x[ind]
w = w[ind]
T = T[ind]
jh = jh[ind]
jl = jl[ind]
jl2 = jl2[ind]
jqp = jqp[ind]
jqp_ = np.sum(jqp,axis=1)

d_eq0 = r['d_eq'][:,0,0]
print('eta', r['eta'])
print('mean duration:', np.mean(durations))

#%%
plt.figure('j3')
plt.clf()
plt.ion()
c = 1/9
magn = 20
plt.plot(x, np.abs(jh)*c, '.-', label='Higgs')
plt.plot(x, np.abs(jl)*magn, label='Leggett')
plt.plot(x, np.abs(jqp_)*c/2, label='QP')

plt.plot(x, np.abs(jh*c-jqp_*c+jl), label='Full')

if omega:
    plt.axvline(d_eq0[0], c='gray', lw=0.5)
    plt.axvline(d_eq0[1], c='gray', lw=0.5)

plt.xlabel(xlabel)

plt.legend()

compare = True
if 'xx' in globals() and compare:
    plt.plot(xx,JQP*factor, '--', c='r')
    plt.plot(xx,JH*factor, '--', c='b')
    plt.plot(xx,JL*factor*magn, '--', c='y')
    plt.plot(xx,(JL+JH+JQP)*factor, '--', c='g')

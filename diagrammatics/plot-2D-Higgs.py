
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
folder = 'j3_g10_5_vall-highres'
folder = 'j3_T'
folder = 'j3_Higgs'
# folder = 'j3_g10_5_vall'
files = glob.glob(f'{folder}/*.pickle')

save_plots = False

plt.ion()

u_t = 6.58285E-2
u_e = 10
u_conductivity = 881.553 #Ohm^-1 cm-1
meV_to_THz = 0.2417990504024
u_w = u_e*meV_to_THz
u_temp = 116.032

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

#%%

v_list = [0.02,0.05,0.2,0.4]

w_list = np.linspace(0.00001,1,100)
T_list = np.linspace(0.00001,50,100)/u_temp
gammas = [ np.array([10, 0.001]), np.array([0.001, 10]),  np.array([10,5])]


vind = 0
for v in v_list:
    vind += 1
    gind = 0
    for g in gammas:
        gind += 1
        JH2D = np.zeros((len(w_list),len(T_list)), dtype=complex)
        for r in res:
            if r['v']==v and (r['g']==gammas[0]).all():
                def indexof(x, array):
                    return np.where(x==array)[0][0]
                JH2D[indexof(r['w'].real, w_list), indexof(r['T'],T_list)] = r['jH']



        plt.figure('Higgs-2D', figsize=(3,2.8))
        plt.clf()
        plt.ion()
        data = np.abs(np.nan_to_num(JH2D))
        plt.pcolormesh(w_list*u_w,T_list*u_temp,data.T, vmax=np.max(data), cmap='cividis')
        plt.xlabel('$\omega$ (THz)')
        plt.ylabel('$T$ (K)')
        plt.colorbar()
        plt.title(f'$v={v}$')
        # plt.xlim((0,1.4))
        plt.ylim((0,50))
        plt.tight_layout()
        plt.savefig(f'higgs-2d/v{vind}-g{gind}_abs.png', dpi=800)
        plt.savefig(f'higgs-2d/v{vind}-g{gind}_abs.pdf', transparent=True)

        plt.figure('Higgs-2d-phase', figsize=(3,2.8))
        plt.clf()
        plt.ion()
        data2 = np.angle(np.nan_to_num(JH2D))/np.pi
        # data2 -= data2[0,0]
        plt.pcolormesh(w_list*u_w,T_list*u_temp,data2.T, cmap='hsv', vmin=-1, vmax=1)
        plt.xlabel('$\omega$ (THz)')
        plt.ylabel('$T$ (K)')
        plt.colorbar()
        plt.title(f'$v={v}$')
        # plt.xlim((0,1.4))
        plt.ylim((0,50))
        plt.tight_layout()
        plt.savefig(f'higgs-2d/v{vind}-g{gind}_phase.png', dpi=800)
        plt.savefig(f'higgs-2d/v{vind}-g{gind}_phase.pdf', transparent=True)

plt.figure('a')
plt.clf()
# plt.plot(w_list,data[:,0])
def nm(x):
    return x/np.max(x, axis=0)
plt.plot(T_list,nm(data[::5].T))

plt.figure('b')
plt.clf()
# plt.plot(w_list,data[:,0])
plt.plot(T_list,np.mod(data2[::5].T+0.5, 2))
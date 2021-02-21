
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
folder = 'j3_QP'
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


v_list = [0.01,0.02,0.05,0.1,0.2,0.4]
# w_list = np.array([0.3,0.4,0.5,0.6,0.7])
T_list = np.array([0.01, 15, 30, 50])/u_temp
v = 0.02
gammas = [np.array([10, 0.00001]), np.array([0.00001, 10]),  np.array([10,5]), np.array([0.00001,0.00001])]
v_list = [0.02]
gammas = [np.array([10,5])]
T_list = [0.4/u_temp]

ii = 0
for v in v_list:
    ii += 1
    kk = 0
    for T_sel in T_list:
        kk += 1

        w = []
        T = []
        jh = []
        jl = []
        jl2 = []
        jqp = []
        durations = []
        d_eq = []
        for r in res:
            if r['v']==v and r['T']==T_sel:# and (r['g']==gammas[0]).all():
                r2 = r
                w.append(r['w'])
                T.append(r['T'])

                jh.append(r['jH'])
                jl.append(r['jL'])
                jl2.append(r['jL2'])
                jqp.append(r['jQP'])

                d_eq.append(r['d_eq0'])

                durations.append(r['duration'])
        if len(w) == 0:
            continue
        r = r2

        w = np.array(w)
        T = np.array(T)
        jl = np.array(jl)
        jl2 = np.array(jl2)
        jh = np.array(jh)
        jqp = np.stack(jqp)
        d_eq = np.stack(d_eq)

        if len(np.unique(w))>1:
            x = np.copy(w)
            xlabel = '$\omega$'
            omega = True
        else:
            x = np.copy(T)*u_temp
            xlabel = '$T$'
            omega = False
            ww = w[0].real

        ind = np.argsort(x)

        x = x[ind]
        d_eq = d_eq[ind]
        w = w[ind]
        T = T[ind]
        jh = jh[ind]
        jl = jl[ind]
        jl2 = jl2[ind]
        jqp = jqp[ind]
        jqp_ = np.sum(jqp,axis=1)

        d_eq0 = r['d_eq'][:,0,0]
        g = r['g']
        v = r['v']
        Ne = r['Ne']
        eta = r['eta']
        print('eta', r['eta'])
        print('mean duration:', np.mean(durations))
        print('max duration:', np.max(durations))

        #%%
        plt.ion()
        c = 1/9
        c = 1
        overall = 0.5
        overall = 1
        magn = 20


        def vlines():
            if omega:
                plt.axvline(d_eq0[0], c='gray', lw=0.5)
                plt.axvline(d_eq0[1], c='gray', lw=0.5)
            else:
                ind = np.argmin(np.abs(ww-d_eq[:,0]))
                plt.axvline(x[ind], c='gray', lw=0.5)

                ind = np.argmin(np.abs(ww-2*d_eq[:,1]))
                plt.axvline(x[ind], c='gray', lw=0.5)


        plt.figure('j3', figsize=(7,7))
        plt.clf()
        plt.plot(x, overall*np.abs(jh)*c, '.-', label='Higgs')
        plt.plot(x, overall*np.abs(jl)*magn, '.-', label=f'Leggett (x{magn})')
        plt.plot(x, overall*np.abs(jqp_)*c, label='QP')
        # plt.plot(x, np.abs(jh*c-jqp_*c/2+jl), label='Full')
        plt.title(f'$g={g} v={str(v)}$')
        plt.legend()
        vlines()


        # plt.figure('j3-phase', figsize=(9,8))
        # plt.clf()
        # def angle(x):
        #     xx = np.angle(x)/np.pi
        #     add = np.ones(len(x))*2
        #     add[xx>=0]=0
        #     return (xx+add)
        # plt.plot(x, angle(jh), '.-', label='Higgs')
        # plt.plot(x, angle(jl), label='Leggett')
        # plt.plot(x, angle(jqp_), label='QP')
        # plt.title(f'$g={g} v={str(v)}$')
        # plt.ylabel('$\\varphi/\pi$')
        # plt.xlabel(xlabel)
        # plt.legend()
        # plt.tight_layout()
        # # plt.savefig(next('phase'))
        # vlines()




        plt.xlabel(xlabel)

        if omega:
            plt.title(f'$g={g}, v={str(v)}, N_E={Ne}, \eta={eta}, T={np.round(T_sel*u_temp,1)}$')
        else:
            plt.title(f'$g={g}, v={str(v)}, w={np.round(ww*u_w,2)}, N_E={Ne}, \eta={eta}$')
        plt.legend()

        plt.pause(0.01)
        # plt.savefig(next(f'grid-g6-v{ii}-w{kk}'))

        compare = False
        if 'xx' in globals() and compare:
            if omega:
                idx = np.argmin(np.abs(vs-v))
                u = 1
            else:
                idx = np.argmin(np.abs(wss-ww))
                plt.title(f'$g={g}, v={str(v)}, w={np.round(ww*u_w,2)}, N_E={Ne}, \eta={eta}$\n h={h}, l={np.round(l,3)}, q={np.round(q,3)}')
                u = u_temp

            factor = 0.01619978567238627
            factor = np.abs(jh[0])/np.abs(JH_[idx,0])

            h = 1
            l = 1 / (np.abs(JL_)[idx,0]*factor/np.abs(jl[0]))
            # l = 1/4
            # q = 1
            q = 1 / (np.abs(JQP_)[idx,-1]*factor/np.abs(jqp_[-1]))


            # factor = np.max(np.abs(jl))/np.max(np.abs(JL_[idx]))

            plt.figure('j3', figsize=(9,8))
            plt.plot(np.array(xx)*u,np.abs(JH_)[idx]*factor*h, '--', c='b')
            plt.plot(np.array(xx)*u,np.abs(JL_)[idx]*factor*magn*l, '--', c='y')
            plt.plot(np.array(xx)*u,np.abs(JQP_)[idx]*factor*q, '--', c='r')

            # plt.figure('j3-phase', figsize=(9,8))
            # plt.plot(xx,angle(JQP_[idx]), '--', c='r')
            # plt.plot(xx,angle(JH_[idx]), '--', c='b')
            # plt.plot(xx,angle(JL_[idx]), '--', c='y')
            # plt.plot(xx,np.abs(JFULL_[idx])*factor, '--', c='g')


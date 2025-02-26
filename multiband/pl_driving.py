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
from scipy.signal import find_peaks

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

folder = 'leggett-driving4'
folder = 'driving'
files = glob.glob(f'{folder}/*.pickle')

save_plots = False

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
def values(key):
    vals = []
    for r in res:
        entry = r[key]
        if type(entry) == np.ndarray:
            for e in entry:
                vals.append(e)
        else:
            vals.append(entry)
    return list(dict.fromkeys(vals))

def sel(first=False, **kwargs):
    ret = []
    for r in res:
        cond = True
        for key, value in zip(kwargs,kwargs.values()):
            if np.array(r[key]==value).all() == False:
                cond=False
                break
        if cond:
            ret.append(r)
            if first:
                return r
    return ret

def title(r):
    freq = r['w']
    tau = np.round(r['tau'],2)
    T = r['T']
    g = r['g']
    g1 = g[0]
    if len(g)>1:
        g2 = g[1]
    else:
        g2=0
    A0 = r['A0']
    A0_pr = r['A0_pr']
    w_pr = np.round(r['w_pr'],2)
    tau_pr = np.round(r['tau_pr'],2)
    t_delay = np.round(r['t_delay'],2)
    plt.title(f"$\omega={freq}, \\tau={tau}, \gamma_1={g1},$\n $\gamma_2={g2}, T={T}, \Delta t={t_delay}$ \n $A_0={A0}, A_0\'={A0_pr}, \\tau\'={tau_pr}, \omega\'={w_pr}$")

def getE(r, t=None):
    A0 = r['A0']
    A0_pr = r['A0_pr']
    tau = r['tau']
    tau_pr = r['tau_pr']
    w = r['w']
    te = r['te']
    te_pr = r['te_pr']
    w_pr = r['w_pr']
    if (not hasattr(t, "__len__")) and t==None:
        t = r['t']
    t_delay = r['t_delay']

    efield = r['efield']

    def A(t):
        # return A_pump(t) + A_probe(t)
        """ Returns the vector potential at time t """
        return A0*np.exp(-(t-te)**2/(2*tau**2))*np.cos(w*t) \
            +  A0_pr*np.exp(-(t-te-t_delay)**2/(2*tau_pr**2))*np.cos(w_pr*(t-t_delay))

    return A(t), efield

def plotHiggs(r, offset = 0.5, plot=True):
    d = r['d_2'].real
    d_im = r['d_2'].real
    t = r['t']
    tau = r['tau']
    e = r['efield']
    w = r['w']
    d_eq = r['d_eq'][:,0,0]
    g = r['g']
    A = r['A']

    # determine left boundary for fft
    tleft = 3*tau+1
    tright = 140
    cond = np.logical_and(t>=tleft, t<=tright)

    t_ = t[cond]
    d_ = d[cond]
    d_ -= np.mean(d_, axis=0)
    if plot:
        plt.figure('Higgs', figsize=(5,2.5))
        plt.clf()
        plt.subplot(121)
        plt.axvspan(tleft,tright,facecolor='gray', alpha=0.3)
        plt.plot(t,d)
        plt.xlabel(f'$t$')
        plt.ylabel(f'Re$\delta \Delta$')
        # title(r)

    lw = np.copy(g)
    lw[lw>0.01] = 2
    lw[lw<0.01] = 0.8
    w_, dw_ = fft(t_, d_)
    if plot:
        plt.subplot(122)
        plt.axvline(d_eq[0]*2, c='gray', lw=lw[0])
        plt.axvline(d_eq[1]*2, c='gray', lw=lw[1])
        plt.plot(w_, np.abs(dw_))
        plt.xlim((0,4))
        plt.xlabel(f'$\omega$')
        plt.tight_layout()

    return t,d, d_im, w_, np.abs(dw_)

def plotHiggsLast(r, offset=0):
    plt.figure('Higgslast')
    plt.clf()
    d = r['d_2'].real
    t = r['t']
    tau = r['tau']
    e = r['efield']
    w = r['w']
    d_eq = r['d_eq'][:,0,0]
    g = r['g']

    t_ = t[t>offset]
    d_ = d[t>offset]
    d_ -= np.mean(d_, axis=0)
    plt.subplot(121)
    plt.plot(t,d)
    plt.axvline(offset, c='gray', lw=2,ls='--')
    plt.xlabel(f'$t$')
    plt.ylabel(f'Re$\delta \Delta$')
    title(r)

def conductivity(r, order=1, plot=True, begin=None, end=None, subtract=True):
    r2 = sel(A0=r['A0'], g=r['g'], A0_pr=0)[0]
    # r3 = sel(T=0.22, t_delay=r['t_delay'], A0=0, A0_pr=r['A0_pr'])[0]

    if order == 1:
        jd = r['jd_1']
        jp = r['jp_1']
        jd2 = r2['jd_1']
        jp2 = r2['jp_1']
        # jd3 = r3['jd_1']
        # jp3 = r3['jp_1']
        lb='$j_1$'
    else:
        jd = r['jd_3']
        jp = r['jp_3']
        jd2 = r2['jd_3']
        jp2 = r2['jp_3']
        # jd3 = r3['jd_3']
        # jp3 = r3['jp_3']
        lb = '$j_3$'
    j = jd + jp
    j2 = jd2 + jp2
    # j3 = jd3 + jp3
    if subtract:
        j = j - j2
    t = r['t']
    tau = r['tau']
    tau_pr = r['tau_pr']
    freq = r['w']
    # e = r['efield']
    A1, e1 = getE(r)
    A2, e2 = getE(r2)
    if subtract:
        A = A1 - A2
        e = e1 - e2
    else:
        A = A1
        e = e1
    d_eq = r['d_eq'][:,0,0]
    tmin = np.min(t)
    tmax = np.max(t)
    Nt = len(t)
    t_delay = r['t_delay']
    g = r['g']

    T = tmax - tmin

    if begin==None:
        begin = t[0] + t_delay - delays[0] + 1e-9
        end = (t_delay-delays[0]) + t[-1] - (delays[-1] - delays[0]) - 1e-9
    sl = np.logical_and(t>begin, t<end)

    w, jw = fft(t[sl],j[sl])
    w, ew = fft(t[sl],e[sl])
    w, aw = fft(t[sl],A[sl])
    #calculate conductivity
    s=jw/(1j*w*aw)
    s[w==0] = 0 + 1j*np.inf
    #this is here to find the max of the conductivity so the plots could be normalized
    gap = np.max(d_eq)
    subs=s[np.abs(w.real-2*gap)<1.9*gap]
    smax = nmax(subs.real)
    smax2 = nmax(subs.imag)

    if plot:
        plt.figure('sigma')
        plt.clf()
        plt.subplot(121)
        plt.axvline(begin, c='gray', lw=1,ls='--')
        plt.axvline(end, c='gray', lw=1,ls='--')
        title(r)
        plt.plot(t, e/nmax(e[sl])*nmax(j[sl]),label='$E$')
        plt.plot(t,j, label=lb + ' (pp-p0)')
        # plt.plot(t,j3, label=lb + ' (probe only)')
        plt.xlabel(f'$t$')
        plt.legend()
        mm = max(np.abs(min(j[sl])),max(j[sl])) * 1
        plt.ylim((-mm*1.1,mm*1.1))

        plt.subplot(122)
        plt.plot(w.real, np.abs(s.real), label = r'$\Re( \sigma)$')
        plt.plot(w.real, np.abs(s.imag)/smax2*smax, label = r'$\Im( \sigma)$')
        plt.legend()
        lw = np.copy(g)
        lw[lw>0.01] = 2
        lw[lw<0.01] = 0.8
        plt.axvline(d_eq[0]*1, c='gray', lw=lw[0], ls='-.')
        plt.axvline(d_eq[0]*2, c='gray', lw=lw[0], ls='-.')
        plt.axvline(d_eq[0]*3, c='gray', lw=lw[0], ls='-.')
        if len(d_eq)>1:
            plt.axvline(d_eq[1]*1, c='gray', lw=lw[1], ls='-')
            plt.axvline(d_eq[1]*2, c='gray', lw=lw[1])
            plt.axvline(d_eq[1]*3, c='gray', lw=lw[1], ls='-')
        plt.xlim(0,4*gap)
        plt.ylim(0,1.2*smax)
        plt.xlabel(f'$\omega$')
        plt.tight_layout()
    # sl2 = np.logical_and(w>0, w<4*d_eq[0])
    sl2 = np.logical_and(w>0, w<6*gap)
    return w[sl2], s[sl2]

def plotLeggett(r, offset=0):
    d_eq = r['d_eq'][:,0,0]
    d = r['d_2']
    dphase = d[:,0].imag/d_eq[0] - d[:,1].imag/d_eq[1]
    t = r['t']
    tau = r['tau']
    w = r['w']
    g = r['g']
    v = r['v']

    tleft = 3*tau+1
    tright = 140
    cond = np.logical_and(t>tleft, t<=tright)

    t_ = t[cond]
    dp_ = dphase[cond]
    dp_ -= np.mean(dp_, axis=0)
    plt.figure('Leggett', figsize=(5,2.5))
    plt.clf()
    plt.subplot(121)
    plt.ylabel('$\delta\\varphi, v=$'+str(np.round(v,3)))
    plt.axvspan(tleft,tright,facecolor='gray', alpha=0.3)
    plt.plot(t,dphase)
    # plt.ylim((np.min(dp_),np.max(dp_)))
    # plt.ylim((-0.0002,0.0002))
    plt.xlabel(f'$t$')

    plt.subplot(122)
    w_, dpw_ = fft(t_, dp_)
    # plt.axvline(d_eq[0]*2, c='gray', lw=1)
    # plt.axvline(d_eq[1]*2, c='gray', lw=1)

    lw = np.copy(g)
    lw[lw>0.01] = 2
    lw[lw<0.01] = 0.8
    plt.axvline(d_eq[0]*2, c='gray', lw=lw[0], ls='-')
    if len(d_eq)>1:
        plt.axvline(d_eq[1]*2, c='gray', lw=lw[1])
    plt.plot(w_, np.abs(dpw_))
    plt.xlim((0,2))
    plt.xlabel(f'$\omega$')
    plt.tight_layout()

    dpw_[w_<0] = 0
    ileggett = np.argmax(np.abs(dpw_))
    wlegget = w_[ileggett]
    vallegget = np.abs(dpw_[ileggett])
    plt.plot([wlegget], [vallegget], 'rx')

    return wlegget
def plotHiggsLeggett(r):
    d = r['d_2'].real
    d_im = r['d_2'].real
    t = r['t']
    tau = r['tau']
    e = r['efield']
    w = r['w']
    d_eq = r['d_eq'][:,0,0]
    g = r['g']
    A = r['A']
    v = r['v']

    plt.figure('HL',figsize=(6,3.5))
    plt.clf()
    fig, axs = plt.subplots(1,3,num='HL')
    axh = axs[0]
    axl = axs[1]
    ax1 = axs[2]

    alpha = 0.9
    lww = 1


    # determine left boundary for fft
    tleft = 3*tau+1

    tright = 120
    cond = np.logical_and(t>=tleft, t<=tright)

    # axh.set_ylabel('$\delta\\varphi, v=$'+str(np.round(v,3)))
    axh.set_xlabel('$t$')
    axh.axvspan(tleft,tright,facecolor='gray', alpha=0.3)
    axh.plot(t,d.real, alpha=alpha,lw=lww)
    axh.set_xlim((min(t)+10, tright))

    ax2 = ax1.twinx()


    t_ = t[cond]
    d_ = d[cond]
    d_ -= np.mean(d_, axis=0)

    lw = np.copy(g)
    lw[lw>0.01] = 1
    lw[lw<0.01] = 0.3
    w_, dw_ = fft(t_, d_)
    color = 'black'
    ax1.set_xlabel(f'$\omega$')
    ax1.set_ylabel('$\delta\Delta\'$', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(d_eq[0]*2, c='gray', lw=lw[0])
    ax1.axvline(d_eq[1]*2, c='gray', lw=lw[1])
    ax1.axvline(2*w, c='green', lw=1, ls='--')
    ax1.plot(w_, np.abs(dw_), alpha=alpha,lw=lww)

    d = r['d_2']
    dphase = d[:,0].imag/d_eq[0] - d[:,1].imag/d_eq[1]

    dp_ = dphase[cond]
    dp_ -= np.mean(dp_, axis=0)

    w_, dpw_ = fft(t_, dp_)
    # plt.axvline(d_eq[0]*2, c='gray', lw=1)
    # plt.axvline(d_eq[1]*2, c='gray', lw=1)

    color = 'tab:red'
    # ax2.set_ylabel('$\delta\\varphi, v=$'+str(np.round(v,3)), color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(w_, np.abs(dpw_), color=color, alpha=alpha, lw=lww)
    plt.xlim((0,2))


    axl.set_ylabel('$\delta\\varphi, v=$'+str(np.round(v,3)))
    axl.set_xlabel('$t$')
    axl.axvspan(tleft,tright,facecolor='gray', alpha=0.3)
    axl.plot(t,dphase, c='tab:red', lw=0.8, alpha=alpha)
    axl.set_xlim((min(t)+10, tright))

    plt.tight_layout()

    dpw_[w_<0] = 0
    ileggett = np.argmax(np.abs(dpw_))
    wlegget = w_[ileggett]
    vallegget = np.abs(dpw_[ileggett])
    # ax2.plot([wlegget], [vallegget], 'rx')

    return wlegget


def plotE(r):
    plt.figure()
    d = r['d_2'].real
    t = r['t']
    tau = r['tau']
    e = r['efield']
    w = r['w']
    d_eq = r['d_eq'][:,0,0]

    plt.subplot(121)
    plt.plot(t,e)
    plt.ylabel(f'$E$')
    plt.xlabel(f'$t$')
    plt.subplot(122)
    tw, ew = fft(t,e)
    plt.plot(tw, np.abs(ew))
    plt.xlim((0,w + 1/tau*5))
    plt.xlabel(f'$\omega$')
    plt.axvline(d_eq[0], c='gray', lw=1)
    if len(d_eq)>1: plt.axvline(d_eq[1], c='gray', lw=1)
    plt.tight_layout()

def plotA(r):
    d = r['d_2'].real
    t = r['t']
    tp = t
    tau = r['tau']
    e = r['efield']
    wnew = r['w']
    d_eq = r['d_eq'][:,0,0]
    A, e = getE(r)

    def nm(x):
        return x / np.max(np.abs(x))

    plt.figure('A',figsize=(10,2))
    plt.clf()
    plt.subplot(131)
    plt.plot(tp,A)
    plt.xlim((min(tp), abs(min(tp))))
    # plt.xlim((-1,1))
    plt.ylabel(f'$A(t)$')
    plt.xlabel(f'$t$')

    plt.subplot(132)
    tw, aw = fft(t,A)
    # plt.plot(tw, np.real(nm(aw)))
    # plt.plot(tw, np.imag(nm(aw)))
    plt.plot(tw, np.abs(nm(aw)),'-')
    plt.xlim((0,5*d_eq[0]))
    plt.ylabel(f'$A(\omega)$')
    plt.xlabel(f'$\omega$')
    plt.axvspan(2*d_eq[0], 4*d_eq[1], facecolor='gray', alpha=0.3)
    if len(d_eq)>1: plt.axvspan(2*d_eq[1], 4*d_eq[1], facecolor='gray', alpha=0.2)
    plt.xlim((0,3*d_eq[1]))
    plt.axvline(2*d_eq[0], c='k', lw=0.2)
    plt.axvline(2*d_eq[0], c='k', lw=0.2)
    plt.tight_layout()

    plt.subplot(133)
    tw, aw2 = fft(t,A**2)
    # plt.plot(tw, np.real(nm(aw2)))
    # plt.plot(tw, np.imag(nm(aw2)))
    plt.plot(tw, np.abs(nm(aw2)),'-')
    plt.xlim((0,5*d_eq[0]))
    plt.ylabel(f'$A^2(\omega)$')
    plt.xlabel(f'$\omega$')
    plt.axvline(2*d_eq[0], c='gray', lw=1)
    plt.xlim((0,3*d_eq[1]))
    if len(d_eq)>1: plt.axvline(2*d_eq[1], c='gray', lw=1)
    plt.tight_layout()
def cond_pcolor():
    plt.figure('cc', figsize=(3,3))
    plt.clf()
    # plt.subplot(121)
    def nm2(x):
        mean = np.mean(x,axis=0)
        x -= mean
        x /= np.max(np.abs(x), axis=0)
        # return x/(mean[:,np.newaxis])
        # return x/(mean)
        return x

    plt.pcolormesh(w*u_e*meV_to_THz, delays*u_t, nm2(np.abs(cc.real)), cmap=cm, shading='gouraud')#, vmin=0.9, vmax=1.1)
    plt.axvline(2*gap*u_e*meV_to_THz)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('$\delta t_{pp}$ (ps)')
    plt.tight_layout()
    savefig('cond-pcolor-real.pdf', transparent=True)
    # plt.colorbar()

    # plt.subplot(122)
    # plt.pcolormesh(w*u_e*meV_to_THz, delays*u_t, nm2(np.abs(cc.imag)), cmap=cm)#, shading='gouraud')#, vmin=0.9, vmax=1.1)
    # plt.axvline(2*gap*u_e*meV_to_THz)
    # plt.xlabel('Frequency (THz)')
    # plt.ylabel('$\delta t_{pp}$ (ps)')
    # # plt.colorbar()
    # plt.tight_layout()

def cond_3d():
    fig = plt.figure('cc3d', figsize=(3.5,3.5))
    plt.clf()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(w*u_e*meV_to_THz, delays*u_t)
    ax.plot_surface(X, Y, np.abs(cc.real), cmap=cm)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('$\delta t_{pp}$ (ps)')
    ax.ticklabel_format(axis="z", style="sci", scilimits=(2,4))
    ax.set_zlabel('$\sigma\,\'$ ($\Omega^{-1}$cm$^{-1}$)')
    ax.set_xticks([0,0.5,1,1.5])
    plt.tight_layout(pad=2)
    savefig('cond-3d-real.pdf', transparent=True)
    # ax.set_zlim((0,100))

    fig = plt.figure('cc3dim', figsize=(3.3,3.3))
    plt.clf()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(w*u_e*meV_to_THz, delays*u_t)
    ax.plot_surface(X, Y, np.abs(cc.imag), cmap=cm)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('$\delta t_{pp}$ (ps)')
    ax.set_xticks([0,0.5,1,1.5])
    ax.ticklabel_format(axis="z", style="sci", scilimits=(2,4))
    ax.set_zlabel('$\sigma\,\'$ ($\Omega^{-1}$cm$^{-1}$)')
    plt.tight_layout(pad=2)
    savefig('cond-3d-imag.pdf', transparent=True)

def cond_w_pcolor():
    plt.figure('ccw', figsize=(3,3))
    plt.clf()

    def nm3(x):
        mean = np.mean(x,axis=0)
        # x -= mean
        x /= np.max(np.abs(x), axis=0)
        # return x/(mean[:,np.newaxis])
        # return x/(mean)
        return x

    # plt.subplot(121)
    sl3 = np.logical_and(w_delay>=0,w_delay<=gap*4)
    plt.pcolormesh(w*u_e*meV_to_THz, w_delay[sl3]*u_e*meV_to_THz,nm3(np.abs(ccw[sl3])), cmap=cm)#, shading='gouraud')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('$\omega_{\delta t_{pp}}$ (THz)')
    plt.ylabel('$\mathcal{FT}[\delta t_{pp}]$ (THz)')
    plt.tight_layout()
    savefig('cond-pcolor-w-real.pdf', transparent=True)
    # plt.colorbar()

    # plt.subplot(122)

    # plt.pcolormesh(w, w_delay[sl3], nm3(np.abs(ccwi[sl3])), cmap=cm, shading='gouraud')
    # plt.colorbar()
    # plt.xlabel('$\omega$')
    # plt.ylabel('$\omega_{\delta t_{pp}}$')
    # plt.tight_layout()

def cond_w_3d():
    fig = plt.figure('ccw3d')
    ax = fig.gca(projection='3d')
    sl3 = np.logical_and(w_delay>=0,w_delay<=gap*4)
    X, Y = np.meshgrid(w, w_delay[sl3])
    ax.plot_surface(X, Y, np.abs(ccw[sl3]), cmap=cm)
    plt.xlabel('$\omega$')
    plt.ylabel('$\omega_{\delta t_{pp}}$')
    plt.title('Re $\sigma$')

    fig = plt.figure('ccw3dim')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, np.abs(ccwi[sl3]), cmap=cm)
    plt.xlabel('$\omega$')
    plt.ylabel('$\omega_{\delta t_{pp}}$')
    plt.title('Im $\sigma$')

def cond_w_single():
    plt.figure('ccwsingle')
    plt.clf()
    def nm(x):
        return x / np.max(np.abs(x))
    for k in np.arange(10,ccm.shape[1],3):
        y = ccm.real[:,k]
        plt.subplot(121)
        plt.plot(delays, nm(y))
        plt.ylabel('Re $\sigma$')
        plt.xlabel('$\delta t_{pp}$')
        plt.subplot(122)
        www, cw = fft(delays,y)
        plt.plot(www,nm(np.abs(cw)))
        plt.xlabel('$\omega_{\delta t_{pp}}$')
        plt.ylabel('Re $\sigma$')

    plt.xlim((0,5*gap))
    plt.tight_layout()


delays = np.sort(values('t_delay'))
durations = values('duration')
temps = np.sort(values('T'))
A0s = np.sort(values('A0'))
A0_prs = np.sort(values('A0_pr'))
gammas = np.sort(values('g'))
vs = np.sort(values('v'))
r0 = sel(t_delay=delays[-1])[0]
# d2_ref = r0['d_2'].real
d_eq = r0['d_eq'][:,0,0]
gap = np.max(d_eq)
t = r0['t']
taus = np.sort(values('tau'))
wss = np.sort(values('w'))

gammas = [0.001, 10]

cleanclean= np.array([gammas[0], gammas[0]])
cleandirty = np.array([gammas[0], gammas[1]])
dirtyclean = np.array([gammas[1], gammas[0]])
dirtydirty = np.array([gammas[1], gammas[1]])

fourcases = [cleanclean, cleandirty, dirtyclean, dirtydirty]
# fourcases = [cleanclean]
lwc = 0.3
lwd = 0.6
lw_fourcases = [[lwc,lwc],[lwc,lwd],[lwd,lwc],[lwd,lwd]]

u_t = 6.58285E-2
u_e = 10
u_conductivity = 881.553 #Ohm^-1 cm-1
meV_to_THz = 0.2417990504024

#1A are 2.38459E-7 Js/Cm
u_A = 2.38459E-7
A,e = getE(r0)
Amax = np.max(np.abs(A))

############################
A0 = 0.5E-8 # Js/Cm
############################

Ascale = A0 / (Amax * u_A)

cm = 'rainbow'

#%%
def z(array):
    return zip(np.arange(len(array)), array)

vv = []
en = []
for j, v in z(vs):
    plt.pause(0.1)
    for i, case in z(fourcases):
        xx = []
        yy = []
        # plt.figure()
        for k, w in z(wss):
            r = sel(first=False, w=w, v=v, g=case)
            if r == []:
                continue
            if len(r) > 1: print('Error:', len(r), 'matches found')
            r = r[0]
            j1 = r['jd_1'] + r['jp_1']
            j3 = r['jd_3'] + r['jp_3']

            tw, j1w = fft(t,j1)
            tw, j3w = fft(t,j3)

            if k==6:
                plt.figure(figsize=(6,3))
                plt.subplot(121)
                plt.plot(t,nm(j1),label='$j_1$')
                plt.plot(t,nm(j3),label='$j_3$')
                plt.xlabel('t')
                plt.legend()

                plt.subplot(122)
                plt.plot(tw/w, nm(np.abs(j3w)))
                plt.plot(tw/w, np.abs(j3w))
                plt.axvline(1)
                plt.axvline(3)
                plt.axvline(5)
                plt.axvline(7)
                plt.xlim((0,6))
                plt.xlabel('$\omega/\Omega$')

            # y = nm(np.abs(j3w))
            y = np.abs(j3w)
            twn = tw/w
            y[np.abs(twn-3)>0.2] = 0

            xx.append(w)
            yy.append(np.max(y))

        plt.figure(figsize=(3,3))
        plt.plot(xx,yy, '.-')
        plt.title(f'v={np.round(v,2)}, g={case}')
        lw = np.copy(case)
        lw[lw>0.01] = 2
        lw[lw<0.01] = 0.8
        plt.axvline(d_eq[0], c='gray', lw=lw[0])
        plt.axvline(d_eq[1], c='gray', lw=lw[1])

        plt.xlabel('$\omega$ (of driving pulse)')
        plt.ylabel('THG signal strength (a.u.)')

        peaks, _ = find_peaks(yy, width=2, distance=5)
        print(peaks)
        plt.plot(np.array(xx)[peaks],np.array(yy)[peaks],'x')

        if len(peaks) > 0:
            vv.append(v)
            en.append(xx[peaks[0]])


        plt.tight_layout()
        # plt.savefig(f'figs-driving/{folder}-v{j}-case{i}.pdf')


plt.figure()
plt.plot(vv,np.array(en)*2,'.-')
plt.axhline(2*d_eq[0],c='gray',lw=0.5)
plt.axhline(2*d_eq[1],c='gray',lw=0.5)
plt.xlabel('$v$')
plt.ylabel('$\omega$')
# plt.savefig(f'figs-driving/{folder}-legget-energy.pdf')



# #%%
# w = 0.7
# r = sel(first=True, w=w, v=0.02, g=dirtydirty)
# t = r['t']
# j1 = r['jd_1'] + r['jp_1']
# j3 = r['jd_3'] + r['jp_3']
# plt.plot(t,nm(j1))
# plt.plot(t,nm(j3))

# tw, j1w = fft(t,j1)
# tw, j3w = fft(t,j3)

# plt.figure()
# plt.plot(tw/w, nm(np.abs(j1w)))
# plt.plot(tw/w, nm(np.abs(j3w)))
# plt.xlim((0,4))

#%%

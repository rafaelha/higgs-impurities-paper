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

folder = 'conductivity-dirty'
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

    t_ = t[t>offset]
    d_ = d[t>offset]
    d_ -= np.mean(d_, axis=0)
    if plot:
        plt.figure('Higgs')
        plt.clf()
        plt.subplot(121)
        plt.plot(t,d)
        plt.axvline(offset, c='gray', lw=2,ls='--')
        plt.xlabel(f'$t$')
        plt.ylabel(f'Re$\delta \Delta$')
        title(r)

    lw = np.copy(g)
    lw[lw>0.01] = 2
    lw[lw<0.01] = 0.8
    if len(d_eq)>1:
        plt.axvline(d_eq[1]*2, c='gray', lw=lw[1])
    w_, dw_ = fft(t_, d_)
    if plot:
        plt.subplot(122)
        plt.axvline(d_eq[0]*2, c='gray', lw=lw[0])
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

def plotLegett(r, offset=0):
    d_eq = r['d_eq'][:,0,0]
    d = r['d_2']
    dphase = d[:,0].imag/d_eq[0] - d[:,1].imag/d_eq[1]
    t = r['t']
    tau = r['tau']
    w = r['w']
    g = r['g']

    t_ = t[t>offset]
    dp_ = dphase[t>offset]
    dp_ -= np.mean(dp_, axis=0)
    plt.figure('Leggett')
    plt.clf()
    plt.subplot(121)
    plt.plot(t,dphase)
    plt.axvline(offset, c='gray', lw=2,ls='--')
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
    plt.axvline(d_eq[0]*1, c='gray', lw=lw[0], ls='-.')
    plt.axvline(d_eq[0]*2, c='gray', lw=lw[0], ls='-.')
    plt.axvline(d_eq[0]*3, c='gray', lw=lw[0], ls='-.')
    if len(d_eq)>1:
        plt.axvline(d_eq[1]*1, c='gray', lw=lw[1], ls='-')
        plt.axvline(d_eq[1]*2, c='gray', lw=lw[1])
        plt.axvline(d_eq[1]*3, c='gray', lw=lw[1], ls='-')
    plt.plot(w_, np.abs(dpw_))
    plt.xlim((0,4))
    plt.xlabel(f'$\omega$')
    plt.tight_layout()

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

    plt.figure('A')
    plt.clf()
    plt.subplot(131)
    plt.plot(tp,A)
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
    plt.axvline(d_eq[0], c='gray', lw=1)
    plt.xlim((0,5*d_eq[0]))
    if len(d_eq)>1: plt.axvline(d_eq[1], c='gray', lw=1)
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
    plt.xlim((0,5*d_eq[0]))
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
gammas_all = np.sort(values('g'))
gammas_all = [np.array([g]) for g in gammas_all]
gammas = []
for g in gammas_all:
    if len(sel(g=g)) == 129:
        gammas.append(g)

r0 = sel(t_delay=delays[-1])[0]
d2_ref = r0['d_2'].real
d_eq = r0['d_eq'][:,0,0]
gap = np.max(d_eq)
t = r0['t']

save = False

# cleanclean= np.array([gammas[0], gammas[0]])
# cleandirty = np.array([gammas[0], gammas[1]])
# dirtyclean = np.array([gammas[1], gammas[0]])
# dirtydirty = np.array([gammas[1], gammas[1]])

u_t = 6.58285E-2
u_e = 10
u_conductivity = 881.553 #Ohm^-1 cm-1
meV_to_THz = 0.2417990504024

cm = 'rainbow'
cc1 = []
cc3 = []
rr = []
delays = delays[1:]
s_imp = 8
# s_imp = 3
for d in delays:
    r = sel(first=True, t_delay=d, A0_pr=A0_prs[0], g=gammas[s_imp])
    rr.append(r)
    w, s1 = conductivity(r, order=1, plot=False)
    w, s3 = conductivity(r, order=3, plot=False)
    cc1.append(s1)
    cc3.append(s3)
cc1 = np.stack(cc1)
cc3 = np.stack(cc3)

##%%

#1A are 2.38459E-7 Js/Cm
u_A = 2.38459E-7
A,e = getE(r)
Amax = np.max(np.abs(A))

############################
A0 = 0.5E-8 # Js/Cm
############################

Ascale = A0 / (Amax * u_A)
cc = (cc1 + Ascale**2 * cc3)*u_conductivity

ccm = cc - np.mean(cc,axis=0)
w_delay,ccw = fft(delays, ccm.real)
w_delay,ccwi = fft(delays, ccm.imag)

# cond_3d()
cond_pcolor()
plt.title(f'$\gamma/2\Delta={gammas[s_imp][0]/2/gap}$')
plt.tight_layout()
# cond_w_pcolor()


#%% fig1 E


gammas_ = np.array([g[0] for g in gammas_all])
dmean = []
damp = []
for g in gammas_all:
    r = sel(first=True, t_delay=0, A0_pr=0, g=g)
    tt, gapt, gapt_im, ww, gapw = plotHiggs(r, plot=False)
    fullgap_thz = 2*np.abs((gap+(gapt+1j*gapt_im)*Ascale**2))*u_e*meV_to_THz
    dmean.append(np.mean(fullgap_thz[tt>0]))
    damp.append(np.max(np.abs(fullgap_thz[tt>0]-np.mean(fullgap_thz[tt>0]))))


fig, ax1 = plt.subplots(figsize=(2.8,2.5))
color = 'k'
ax1.set_xlabel('$\gamma/2\Delta$')
ax1.set_ylabel('$\Delta_{\infty}$ (THz)', color=color)
ax1.plot(gammas_/2/gap, dmean, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks([0,0.5,1,1.5,2])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'darkred'
ax2.set_ylabel('Osc. amplitude (THz)', color=color)  # we already handled the x-label with ax1
ax2.plot(gammas_/2/gap, damp, '--', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.xlim((-0.1,2))
fig.tight_layout()  # otherwfigse the right y-label is slightly clipped
savefig('imp-res.pdf')


#%% fig1 C+D

plt.figure('higgs-t', figsize=(4.5,3.2))
plt.clf()
plt.figure('higgs-w', figsize=(2.8  ,2.5))
plt.clf()
gammas_ = np.array([g[0] for g in gammas_all])
for g in np.array(gammas_all)[[15,26,32,34,35,36,37,38,40,43,44]]:
    r = sel(first=True, t_delay=0, A0_pr=0, g=g)
    plt.figure('higgs-t')
    tt, gapt, gapt_im, ww, gapw = plotHiggs(r, plot=False)
    plt.plot(tt*u_t,2*np.abs((gap+(gapt+1j*gapt_im)*Ascale**2))*u_e*meV_to_THz, label=f'$\gamma/2\Delta={str(np.round(g[0]/2/gap,1))}$')
    # sel2 = tt>0
    # plt.loglog(tt[sel2]*u_t,np.abs(gapt[sel2]-np.mean(gapt[sel2])))
    # plt.plot(t, 300*t**(-1/2))
    plt.ylabel('$|2\Delta(t)| $ (THz)')
    plt.xlabel('$t$ (ps)')
    plt.tight_layout()

    plt.figure('higgs-w')
    plt.plot(ww*u_e*meV_to_THz,nm(gapw), label=f'$\gamma/2\Delta={str(np.round(g[0]/2/gap,1))}$')
    plt.xlim((0.4,0.8))
    plt.ylabel('$\delta\Delta\'(\omega) (a.u.)$')
    plt.xlabel('Frequency (THz)')
    plt.tight_layout()
    # plt.legend()


plt.figure('higgs-t')
plt.xlim((-4.5,20))
plt.ylim((0.4,0.67))
plt.text(3,0.417,'$\gamma/2\Delta=0.5$')
plt.text(3,0.61,'$\gamma/2\Delta=20$')
# plt.legend()
savefig('higgs-t.pdf')
plt.figure('higgs-w')
plt.axvline(2*gap*u_e*meV_to_THz, c='k', lw=0.3)
savefig('higgs-w.pdf', transparent=True)

#%% fig2

plt.figure('cond-imp-real', figsize=(3,2.8))
plt.clf()
plt.figure('cond-imp-imag', figsize=(3,2.8))
plt.clf()
plt.figure('cond-imp-legend', figsize=(5,5))
plt.clf()
gammas_ = np.array([g[0] for g in gammas_all])
# for g in np.array(gammas_all)[[15,26,32,34,35,36,37,38,40,43,44]]:
for g in np.array(gammas_all)[[6,7,11,15,27,35,38,43]]:
    r = sel(first=True, t_delay=0, A0_pr=0, g=g)
    wg, c1 = conductivity(r, order=1, plot=False, subtract=False, begin=t[0], end=t[-1])
    wg, c3 = conductivity(r, order=3, plot=False, subtract=False, begin=t[0], end=t[-1])
    c = (c1 + Ascale**2*c3*0)*u_conductivity
    plt.figure('cond-imp-real')
    if g[0]/2/gap >= 1:
        ls='--'
    else:
        ls='-'
    plt.plot(wg*u_e*meV_to_THz,np.abs(c.real), label=f'$\gamma/2\Delta={str(np.round(g[0]/2/gap,1))}$',ls=ls)
    plt.xlabel('Frequency (THz)')
    plt.xlim((0,4*gap*u_e*meV_to_THz))
    plt.ylim((0,8e4))
    plt.ylabel('$\sigma\,\'$ ($\Omega^{-1}$cm$^{-1}$)')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(2,4))
    # plt.legend()

    plt.figure('cond-imp-imag')
    plt.plot(wg*u_e*meV_to_THz,np.abs(c.imag), label=f'$\gamma/2\Delta={str(np.round(g[0]/2/gap,1))}$',ls=ls)
    plt.xlabel('Frequency (THz)')
    plt.xlim((0,4*gap*u_e*meV_to_THz))
    plt.ylim((0,100e4))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(2,4))
    plt.ylabel('$\sigma\,\'\'$ ($\Omega^{-1}$cm$^{-1}$)')
    # plt.legend()

    plt.figure('cond-imp-legend')
    plt.plot(wg*u_e*meV_to_THz,np.abs(c.imag), label=f'$\gamma/2\Delta={str(np.round(g[0]/2/gap,1))}$',ls=ls)
    plt.ylim((0,0.1))
    plt.legend()

plt.figure('cond-imp-real')
plt.axvline(2*gap*u_e*meV_to_THz, c='k', lw=0.3)
plt.tight_layout()
savefig('cond-imp-real.pdf')

plt.figure('cond-imp-imag')
plt.axvline(2*gap*u_e*meV_to_THz, c='k', lw=0.3)
plt.tight_layout()
savefig('cond-imp-imag.pdf')

plt.figure('cond-imp-legend')
savefig('cond-imp-legend.pdf')
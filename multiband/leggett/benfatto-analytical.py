import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
from scipy import integrate
from scipy import interpolate
from scipy.fftpack import fftfreq, fftshift
import scipy
import time
import pickle
import sys
import os
import gc

A0 = 0.2
w = 0.3
tau = 1
te = 0

Ne = 1000

tmin = -10
tmax = 500
Nt = 450

nb = 2
u_temp = 116.032
T = 0.001/u_temp
wd = 5
s = np.array([1,-1])
m = np.array([0.85, 1.38])
ef = np.array([290, 70])
pre_d0 = np.array([0.3,0.7])

#Constants
hbar=1
kb=1
e_charge=1

B = 1/(kb*T)
kf = np.sqrt(2*m*ef)/hbar
vf = kf/m
n = kf**3/(3*np.pi**2)
N0 = m*kf/(2*np.pi**2)


e = np.linspace(-wd,wd,Ne)

def d0_integrand(x, d):
    # This is an auxiliary function used in find_d0 to calculate an integral
    return 0.5*1/np.sqrt(x**2+d**2)*np.tanh(B/2*np.sqrt(x**2+d**2))


def find_d02(U):
    delta_guess = np.array([1,1])
    while True:
        integral = np.zeros(2)
        for j in [0, 1]:
            integral[j] = integrate.quad(d0_integrand, -wd, wd, (delta_guess[j],))[0]
        dd = U@(N0*delta_guess*integral)
        if np.sum(np.abs(dd-delta_guess)) < 1e-15:
            return dd
        delta_guess = dd

def find_U2(d0, v):
    I = np.zeros(2)
    for j in [0, 1]:
        I[j] = integrate.quad(d0_integrand, -wd, wd, (d0[j],))[0]

    U11 = d0[0]/(N0[0]*d0[0]*I[0]+v*N0[1]*d0[1]*I[1])
    U22 = (d0[1]-v*U11*N0[0]*d0[0]*I[0])/(N0[1]*I[1]*d0[1])
    U12 = v*U11
    U = np.array([[U11, U12],
                    [U12, U22]])
    return U

def nm(x):
    return x / np.max(np.abs(x))

def rfft(t,f, inverse=False):
    # a different implementatation of the fourier transform,
    # I checked that both versions agree with each other
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


def integ(x, axis):
    """ Integrate the function 'x' over the axis 'axis'. The integration can be performed over one or two dimensions """
    if hasattr(axis, "__len__"):
        return integrate.simps(integrate.simps(x, dx=de, axis=axis[1]), dx=de, axis=axis[0])
    else:
        return integrate.simps(x, dx=de, axis=axis)

def plotA(t,A):
    plt.figure('A')
    plt.clf()
    tp = t
    plt.clf()
    plt.subplot(131)
    plt.plot(tp,A)
    # plt.plot(tp,efield)
    plt.ylabel(f'$A(t)$')
    plt.xlabel(f'$t$')

    plt.subplot(132)
    tw, aw = rfft(t,A)
    # plt.plot(tw, np.real(nm(aw)))
    # plt.plot(tw, np.imag(nm(aw)))
    plt.plot(tw, np.abs(nm(aw)),'-')
    plt.xlim((0,5*d_eq0[0]))
    plt.ylabel(f'$A(\omega)$')
    plt.xlabel(f'$\omega$')
    plt.axvline(2*d_eq[0], c='gray', lw=1)
    plt.xlim((0,4*d_eq[1]))
    # if len(d_eq)>1: plt.axvline(d_eq[1], c='gray', lw=1)
    plt.tight_layout()

    plt.subplot(133)
    tw, aw2 = rfft(t,A**2)
    # plt.plot(tw, np.real(nm(aw2)))
    # plt.plot(tw, np.imag(nm(aw2)))
    plt.plot(tw, np.abs(nm(aw2)),'-')
    plt.xlim((0,5*d_eq0[0]))
    plt.ylabel(f'$A^2(\omega)$')
    plt.xlabel(f'$\omega$')
    plt.axvline(2*d_eq[0], c='gray', lw=1)
    plt.xlim((0,4*d_eq[1]))
    if len(d_eq)>1: plt.axvline(2*d_eq[1], c='gray', lw=1)
    plt.tight_layout()

def A(t):
    return A0*np.exp(-(t-te)**2/(2*tau**2))*np.cos(w*t)


v_leggett = 0.1
chis = []
vs = np.linspace(0,1,300)
il = 0
for v_leggett in vs:
    print(il)
    il += 1
    B = 1/(kb*0.000001)
    ep = np.linspace(-wd, wd, Ne)
    U = find_U2(pre_d0,v_leggett)
    UN0 = U*N0[:, np.newaxis]
    print('U=',U)
    print('UN0=',UN0)
    # d_eq0_T0 = find_d0(UN0)
    # print('gap=',d_eq0_T0, 'at T=0 (computed with old function)')
    d_eq0_T0 = find_d02(U)
    print('gap=',d_eq0_T0, 'at T=0 (computed with new function)')
    B = 1/(kb*T)
    # d_eq0 = find_d0(UN0)
    # print('gap=',d_eq0, 'at T=',T, ' (computed with old function)')
    d_eq0 = find_d02(U)
    print('gap=',d_eq0, 'at T=',T, ' (computed with new function)')
    N = N0

    def ref_less(x):
        return 2 / (x*sqrt(1-x**2)) * np.arctan(x/sqrt(1-x**2))
    def ref_greater(x):
        return - 1 / (x*sqrt(x**2-1)) * np.log( (x+sqrt(x**2-1)) / (x-sqrt(x**2-1)))\
        + 1j* np.pi / (x*sqrt(x**2-1))
    def F(w,i):
        d = d_eq0[i]
        x = np.array(w/(2*d),dtype=complex)
        return np.piecewise(x, [x<1,x>=1], [ref_less, ref_greater])

    k = 8*d_eq0[0]*d_eq0[1]*U[0,1]/np.linalg.det(U)
    def F_L(w):
        return k*(N[0]*F(w,0)+N[1]*F(w,1)) / (N[0]*F(w,0)*N[1]*F(w,1))

    def nm(x, axis=-1):
        return x/np.max(x, axis=axis)
    w = np.linspace(0.0001,1.2,800)
    w_ = w + 1j*0.01*d_eq0[0]
    chi = 1/(w_**2-F_L(w))
    chis.append(chi)
    # plt.figure('leggett')
    # plt.clf()

    # plt.plot(w, nm(np.abs(np.real(chi))), label='$\\chi\'$')
    # plt.plot(w, nm(np.abs(np.imag(chi))), label='$\\chi\'\' $')
    # plt.plot(w, nm(np.abs(chi)), label='$|\\chi| $')

    # plt.axvline(sqrt(k*(N[0]+N[1])/(2*N[0]*N[1])))
    # plt.axvline(2*d_eq0[0], c='r')
    # plt.axvline(2*d_eq0[1], c='r')
    # plt.plot(w, nm(np.abs(1/chi)), label='$\omega^2-F_L(\omega)$')
    # plt.legend()
    # plt.xlim((0,2))
    # plt.title(f'k={k}')

#%%
plt.figure('cd',figsize=(5,2.7))
plt.clf()
def nm(x, axis=-1):
    return x/np.max(x, axis=axis)
pc = nm(np.abs(chis).T,axis=0)
plt.pcolormesh(vs,w_,pc, cmap='Blues')

peaks_analytical = np.abs(w_[np.argmax(pc,axis=0)])
plt.axhline(2*d_eq0[0], c='gray', lw=1.1, ls='-')
plt.ylim((0,1.2))
plt.xlabel('$v$')
plt.ylabel('$\omega$')

job_ID = 0
task_ID = 0
dphases = []
vsn = []

reader = open(f'{job_ID}_{task_ID}.pickle','rb')
try:
    while True:
        a = pickle.load(reader)
        dphases.append(a['dphase'])
        vsn.append(a['v'])
except:
    reader.close()

t = a['t']
temp = a['T']
dphases = np.stack(dphases)
vsn = np.array(vsn)

tc = 0
sel = (t>tc)
dp_ = dphases[:,sel]
t_ = t[sel]
w_n, dpw_ = rfft(t_, dp_.T)
# plt.figure('1')

def nm(x):
    return x/np.max(x,axis=0)

# plt.figure('1')
# plt.clf()
# plt.pcolormesh(vs,w_,nm(np.abs(dpw_)))#, vmin=0.5, vmax=0.50000000001)
# plt.axhline(2*d_eq0[0], c='r')

wg = w>=2*d_eq0[0]
mm = np.min(np.abs(pc[wg]-0.5),axis=0)
FWHM1 = np.abs(w_[wg][np.argmin(np.abs(pc[wg]-0.5),axis=0)])
selm = mm<0.1
plt.plot(vs[selm],FWHM1[selm],c='g', ls='--', lw=1.5)

wg = w<=2*d_eq0[0]
mm = np.min(np.abs(pc[wg]-0.5),axis=0)
FWHM2 = np.abs(w_[wg][np.argmin(np.abs(pc[wg]-0.5),axis=0)])
selm = mm<0.1
plt.plot(vs[selm],FWHM2[selm],c='g', ls='--', lw=1.5)

plt.plot(vs,peaks_analytical,c='g', lw=1.5)
peaks_numerical = np.abs(w_n[np.argmax(np.abs(dpw_),axis=0)])
plt.plot(vsn[vsn<0.24],peaks_numerical[vsn<0.24],'D', markersize=3, c='red')
plt.plot(vsn[vsn>0.24][::3],peaks_numerical[vsn>0.24][::3],'D', markersize=3, c='red')
# plt.title(f'T={temp*u_temp}K')
plt.colorbar()
plt.text(0.01,0.62,'$2\Delta_1$',fontsize=9)
plt.text(0.45,1,'FWHM',fontsize=8)
plt.text(0.45,0.766,'$\omega_L$',fontsize=9, color='w')
plt.tight_layout()
# plt.legend('a','b')
plt.savefig('legget-dispersion.png', dpi=600)
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
v_leggett = 0.0

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



ax = np.newaxis

d_eq1 = d_eq0[:,ax]
d_eq = d_eq0[:,ax, ax]
s1 = s[:,ax]
m1 = m[:,ax]
vf1 = vf[:,ax]

t = np.linspace(tmin,tmax,Nt)
plotA(t,A(t))

e1_ = np.linspace(-wd, wd, Ne)

de = e1_[1] - e1_[0]
de2 = de**2

e1 = e1_[ax,:]
e_ = e1_[:,ax]
ep_ = e1_[ax,:]

e = e1_[ax,:,ax]
ep = e1_[ax,ax,:]

E1 = np.sqrt(e1**2 + d_eq1**2)
E = np.sqrt(e**2 + d_eq**2)
Ep = np.sqrt(ep**2 + d_eq**2)

b = np.zeros((3,nb,Ne))
b[2] = 2*e1

s0 = np.zeros((3,nb,Ne))
s0[0] = d_eq1.real / 2 / E1 * np.tanh(E1 / (2 * kb * T))
s0[1] = -d_eq1.imag / 2 / E1 * np.tanh(E1 / (2 * kb * T))
s0[2] = -e1 / 2 / E1 * np.tanh(E1 / (2 * kb * T))

s0 = s0.reshape((3*nb*Ne,))

def ds(t, s):
    s_ = np.copy(s).reshape(3,nb,Ne)

    delta = U @ (N0 * integ(s_[0] - 1j * s_[1], axis=1))
    b[0] = -2*delta[:,ax].real
    b[1] = 2*delta[:,ax].imag
    b[2] = 2*e1 + s1 * A(t)[ax,ax]**2 / (2*m1) #* (1+e1/np.max(e1)*0.3)

    ds_ = np.cross(b,s_,axisa=0,axisb=0,axisc=0).reshape((3*nb*Ne,))
    return ds_



t = np.linspace(tmin,tmax,Nt)

# the built in integrator solves for the r values numerically:
sols = integrate.solve_ivp(ds, (tmin, tmax), s0, t_eval=t)

# extracting the solutions from the solver output:
Y = sols.y.reshape(3, nb, Ne, len(sols.t))
t = sols.t

d = np.einsum('ij,jt->ti',U,N0[:,ax]*integ(Y[0] - 1j * Y[1], axis=1))

#%%
tc_ = 0
tc = tc_

plt.figure('delta-real')
plt.clf()
plt.subplot(121)
plt.plot(t,d.real)
plt.axvspan(tc, np.max(t), facecolor='gray', alpha=0.2)
sel = (t>tc)
dp_ = d[sel].real
t_ = t[sel]
dp_ -= np.mean(dp_, axis=0)
plt.xlabel(f'$t$')
plt.ylabel('Imag$[\delta\Delta]$')
plt.subplot(122)
w_, dpw_ = rfft(t_, dp_)
plt.axvline(d_eq[0]*2, c='gray', lw=1)
plt.axvline(d_eq[1]*2, c='gray', lw=1)
plt.plot(w_, np.abs(dpw_))
plt.xlim((0,4*d_eq[1]))
plt.xlabel(f'$\omega$')
plt.ylabel('Imag$[\delta\Delta]$')
plt.tight_layout()
plt.pause(0.01)

plt.figure('delta')
plt.clf()
plt.subplot(121)
plt.plot(t,d.imag)
plt.axvspan(tc, np.max(t), facecolor='gray', alpha=0.2)
sel = (t>tc)
dp_ = d[sel].imag
t_ = t[sel]
dp_ -= np.mean(dp_, axis=0)
plt.xlabel(f'$t$')
plt.ylabel('Imag$[\delta\Delta]$')
plt.subplot(122)
w_, dpw_ = rfft(t_, dp_)
plt.axvline(d_eq[0]*2, c='gray', lw=1)
plt.axvline(d_eq[1]*2, c='gray', lw=1)
plt.plot(w_, np.abs(dpw_))
plt.xlim((0,4*d_eq[1]))
plt.xlabel(f'$\omega$')
plt.ylabel('Imag$[\delta\Delta]$')
plt.tight_layout()
plt.pause(0.01)

plt.figure('Legget')
plt.clf()
plt.subplot(121)
dp = np.angle(d[:,0]) - np.angle(d[:,1])
plt.plot(t,dp)
plt.axvspan(tc, np.max(t), facecolor='gray', alpha=0.2)
sel = (t>tc)
dp_ = dp[sel]
t_ = t[sel]
dp_ -= np.mean(dp_, axis=0)
plt.xlabel(f'$t$')
plt.ylabel('Imag$[\delta\Delta]$')
plt.subplot(122)
w_, dpw_ = rfft(t_, dp_)
plt.axvline(d_eq[0]*2, c='gray', lw=1)
plt.axvline(d_eq[1]*2, c='gray', lw=1)
plt.plot(w_, np.abs(dpw_))
plt.xlim((0,4*d_eq[1]))
plt.xlabel(f'$\omega$')
plt.ylabel('Imag$[\delta\Delta]$')
plt.tight_layout()
plt.pause(0.01)
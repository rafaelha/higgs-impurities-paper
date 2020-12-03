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

Ne = 3000

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
g = np.array([10,5])

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

ne_ = Ne # number of equations per band
ne = nb * ne_ # number of equations

ax = np.newaxis

d_eq0 = pre_d0

d_eq1 = d_eq0[:,ax]
d_eq = d_eq0[:,ax,ax]
d = d_eq0[:,ax, ax,ax]
s1 = s[:,ax]
m1 = m[:,ax]
vf1 = vf[:,ax]

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

W = 1/np.pi*g[:,ax,ax]/((e-ep)**2+g[:,ax,ax]**2)


def genE(dim):
    if dim == 3:
        ek = e1_[ax,:,ax,ax]
        ek2 = e1_[ax,ax,:,ax]
        ek3 = e1_[ax,ax,ax,:]

        eek = E1[:,:,ax,ax]
        eek2 = E1[:,ax,:,ax]
        eek3 = E1[:,ax,ax,:]

        d = d_eq0[:,ax,ax,ax]

        W12 = W[:,:,:,ax]
        W13 = W[:,:,ax,:]
        W23 = W[:,ax,:,:]

        nfeek = 1.0/(np.exp(B*eek) + 1)
        nfeek2 = 1.0/(np.exp(B*eek2) + 1)
        nfeek3 = 1.0/(np.exp(B*eek3) + 1)
        nfeekm = 1.0/(np.exp(-B*eek) + 1)
        nfeek2m = 1.0/(np.exp(-B*eek2) + 1)
        nfeek3m = 1.0/(np.exp(-B*eek3) + 1)

        fk = 1.0/(np.exp(B*ek) + 1)
        fk2 = 1.0/(np.exp(B*ek2) + 1)
        fk3 = 1.0/(np.exp(B*ek3) + 1)

        return ek,ek2,ek3,eek,eek2,eek3,d,W12,W13,W23,nfeek,nfeek2,nfeek3,nfeekm,nfeek2m,nfeek3m,fk,fk2,fk3

    elif dim == 2:
        ek = e1_[ax,:,ax]
        ek2 = e1_[ax,ax,:]

        eek = E1[:,:,ax]
        eek2 = E1[:,ax,:]

        d = d_eq0[:,ax,ax]

        W12 = W[:,:,:]

        nfeek = 1.0/(np.exp(B*eek) + 1)
        nfeek2 = 1.0/(np.exp(B*eek2) + 1)
        nfeekm = 1.0/(np.exp(-B*eek) + 1)
        nfeek2m = 1.0/(np.exp(-B*eek2) + 1)

        fk = 1.0/(np.exp(B*ek) + 1)
        fk2 = 1.0/(np.exp(B*ek2) + 1)

        return ek,ek2,eek,eek2,d,W12,nfeek,nfeek2,nfeekm,nfeek2m,fk,fk2

    elif dim == 1:
        ek = e1_[ax,:]
        eek = E1
        d = d_eq0[:,ax]

        nfeek = 1.0/(np.exp(B*eek) + 1)
        nfeekm = 1.0/(np.exp(-B*eek) + 1)

        return ek,eek,d,nfeek,nfeekm



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
        if len(axis) == 2:
            return integrate.simps(integrate.simps(x, dx=de, axis=axis[1]), dx=de, axis=axis[0])
        elif len(axis) == 3:
            return integrate.simps(integrate.simps(integrate.simps(x, dx=de, axis=axis[2]), dx=de, axis=axis[1]), dx=de, axis=axis[0])
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

Us = []
d_eq0s = []
empty = True

#%%
v_leggett = 0.1
chis = []
chi_Higgs = []
chi_Leggett = []
vs = np.linspace(0,1,200)

il = -1
for v_leggett in vs:
    il += 1
    print(il)
    #### find U paramerters
    if empty:
        B = 1/(kb*0.000001)
        ep = np.linspace(-wd, wd, Ne)
        U = find_U2(pre_d0,v_leggett)
        Us.append(U)
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
        d_eq0s.append(d_eq0)
        print('gap=',d_eq0, 'at T=',T, ' (computed with new function)')
        N = N0
    else:
        U = Us[il]
        UN0 = U*N0[:, np.newaxis]
        B = 1/(kb*T)
        d_eq0 = d_eq0s[il]
        N = N0

    #### Leggett mode
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
empty = False

u_t = 6.58285E-2
u_e = 10
u_conductivity = 881.553 #Ohm^-1 cm-1
meV_to_THz = 0.2417990504024
u_w = u_e*meV_to_THz

plt.figure('cd',figsize=(5,2.7))
plt.clf()
def nm(x, axis=-1):
    return x/np.max(x, axis=axis)
pc = nm(np.abs(chis).T,axis=0)
plt.pcolormesh(vs,w_*u_w,pc, cmap='Blues')

peaks_analytical = np.abs(w_[np.argmax(pc,axis=0)])
plt.axhline(2*d_eq0[0]*u_w, c='gray', lw=1.1, ls='-')
plt.ylim((0,1.2*u_w))
plt.xlabel('$v$')
plt.ylabel('$\omega$')

job_ID = 0
task_ID = 0
dphases = []
vsn = []

reader = open(f'../multiband/leggett/{job_ID}_{task_ID}.pickle','rb')
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
plt.plot(vs[selm],FWHM1[selm]*u_w,c='g', ls='--', lw=1.5)

wg = w<=2*d_eq0[0]
mm = np.min(np.abs(pc[wg]-0.5),axis=0)
FWHM2 = np.abs(w_[wg][np.argmin(np.abs(pc[wg]-0.5),axis=0)])
selm = mm<0.1
plt.plot(vs[selm],FWHM2[selm]*u_w,c='g', ls='--', lw=1.5)

plt.plot(vs,peaks_analytical*u_w,c='g', lw=1.5)
peaks_numerical = np.abs(w_n[np.argmax(np.abs(dpw_),axis=0)])
plt.plot(vsn[vsn<0.24],peaks_numerical[vsn<0.24]*u_w,'D', markersize=3, c='red')
plt.plot(vsn[vsn>0.24][::3],peaks_numerical[vsn>0.24][::3]*u_w,'D', markersize=3, c='red')
# plt.title(f'T={temp*u_temp}K')
plt.colorbar()
plt.text(0.01,0.62*u_w,'$2\Delta_1$',fontsize=9)
plt.text(0.45,1*u_w,'FWHM',fontsize=8)
plt.text(0.45,0.766*u_w,'$\omega_L$',fontsize=9, color='w')
plt.ylabel('$\omega$ (THz)')
plt.tight_layout()
# plt.legend('a','b')
# plt.savefig('legget-dispersion.png', dpi=600)

#%% Higgs



dim = 2
ek,ek2,eek,eek2,d,W12,nfeek,nfeek2,nfeekm,nfeek2m,fk,fk2 = genE(dim)

ws = np.linspace(0.01,2,500)
eta = 0.004
w = ws[ax,ax,:] + 1j*eta

x11_ = (-2*(-d**2 + eek**2 + ek**2))/(4*eek**3 - eek*w**2)
x11 = integ(x11_, axis=1)

il = -1
det_higgs = []

for v_leggett in vs:
    il += 1
    #### find U paramerters
    U = Us[il]
    UN0 = U*N0[:, np.newaxis]
    B = 1/(kb*T)
    d_eq0 = d_eq0s[il]
    N = N0
    dU = U[0,0]*U[1,1]-U[0,1]**2
    det = (x11[0]+2*U[1,1]/dU)*(x11[1]+2*U[0,0]/dU) - (2*U[0,1]/dU)**2

    det_higgs.append(det)

plt.figure('chi-higgs',figsize=(4.9,2.7))
plt.clf()
det_higgs = np.stack(det_higgs)

# plt.plot(ws, np.real(x11.T) + np.array([2/U[0,0], 2/U[1,1]]))
# plt.plot(ws,np.abs(det.real))
# plt.plot(ws,np.abs(det.imag))
ww = ws*u_w

data = 1/np.abs(det_higgs)
data = np.nan_to_num(data)

sep = (d_eq0[0]+d_eq0[1])/2*u_w*2
data_bottom = np.copy(data)
data_bottom[:,ww>sep]=0
data_bottom = nm(data_bottom.T)

data_top= np.copy(data)
data_top[:,ww<=sep]=0
data_top = nm(data_top.T)

data = nm(data.T)
# data = data.T

# data = data_top+data_bottom




plt.pcolormesh(vs,ws*u_w,np.log(data), cmap='Blues', vmin=-1.5,vmax=-0.5)
plt.colorbar()
plt.ylabel('$\omega$ (THz)')
plt.xlabel('v')

p1 = np.argmax(data[ww<2.3], axis=0)
plt.plot(vs, ww[p1], 'r', lw=0.5)

ww = ws*u_w
data2 = np.copy(data)
data2[ww<2.3] = 0
p2 = np.argmax(data2, axis=0)
plt.plot(vs, ww[p2], 'r', lw=0.5)




plt.tight_layout()
# plt.plot(ws,1/np.abs(det))

# plt.axvline(2*d_eq0[0], c='gray', lw=0.5)
# plt.axvline(2*d_eq0[1], c='gray', lw=0.5)
# plt.plot(ws,chi.real)
# plt.plot(ws,chi.imag)
# plt.savefig('higgs-kernel.png', dpi=600)


#%% Leggett 2


dim = 2
ek,ek2,eek,eek2,d,W12,nfeek,nfeek2,nfeekm,nfeek2m,fk,fk2 = genE(dim)

ws = np.linspace(0.0,1.3,500)
eta = 0.006
w = ws[ax,ax,:] + 1j*eta
w2 = ws+ 1j*eta

x33_ = (-2*(d**2 + eek**2 - ek**2))/(4*eek**3 - eek*w**2)
x33 = N0[:,ax] * integ(x33_, axis=1)

il = -1
det_l= []

for v_leggett in vs:
    il += 1
    #### find U paramerters
    U = Us[il]
    UN0 = U*N0[:, np.newaxis]
    B = 1/(kb*T)
    d_eq0 = d_eq0s[il]
    N = N0
    k = 8*d_eq0[0]*d_eq0[1]*U[0,1]/np.linalg.det(U)

    # det = (-x33[0]-k/w2**2)*(-x33[1]-k/w2**2) - k**2/w2**4
    det = (-x33[0]-x33[1]-4*k/w2**2)*(-x33[0]-x33[1]) - (-x33[0]+x33[1])**2
    det *= w2**2/(x33[0]+x33[1])
    # det = w2**2 + k*(x33[0]+x33[1])/(x33[0]*x33[1])

    det_l.append(det)

plt.figure('chi-leggett',figsize=(4.9,2.7))
plt.clf()
det_l = np.stack(det_l)

# plt.plot(ws, np.real(x11.T) + np.array([2/U[0,0], 2/U[1,1]]))
# plt.plot(ws,np.abs(det.real))
# plt.plot(ws,np.abs(det.imag))
ww = ws*u_w

data = 1/np.abs(det_l)
# data = np.nan_to_num(data)

data = nm(data.T)
# data = data.T


plt.pcolormesh(vs,ws*u_w,data, cmap='Blues')#, vmin=-1.5,vmax=-0.5)
plt.colorbar()
plt.ylabel('$\omega$ (THz)')
plt.xlabel('v')

p1 = np.argmax(data[ww<3], axis=0)
plt.plot(vs, ww[p1], 'r', lw=1)



plt.tight_layout()
# plt.plot(ws,1/np.abs(det))

# plt.axvline(2*d_eq0[0], c='gray', lw=0.5)
# plt.axvline(2*d_eq0[1], c='gray', lw=0.5)
# plt.plot(ws,chi.real)
# plt.plot(ws,chi.imag)
# plt.savefig('higgs-kernel.png', dpi=600)

plt.plot(vsn,peaks_numerical*u_w,'D', markersize=3, c='red')
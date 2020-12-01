#%%
import numpy as np
from numpy import sqrt
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
from scipy.fftpack import fftfreq, fftshift
import scipy
import time
import pickle
import sys
import os
import gc

code_version = {
    "1.0": "first version",
    "1.1": "pulse shape modified, now parameter te, te_pr",
    "1.2": "U calculated from gap and v (gap and v are input parameters), r21e r21o saved"
}

job_ID = -1
task_ID = job_ID
task_count = 0


u_t = 6.58285E-2
u_e = 10
u_conductivity = 881.553 #Ohm^-1 cm-1
meV_to_THz = 0.2417990504024
u_w = u_e*meV_to_THz

u_temp = 116.032
params_ = [
    {
        "Ne": [1200],
        "tmin": [-80],
        "tmax": [300],
        # "tmax": [450],
        "Nt": 450,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([0.3,2])],
        "pre_d0": np.array([0.3,0.7]),
        # "pre_d0": np.array([0.29817841,0.70076507]),
        "v": [0],#np.linspace(0.0002,1,40),
        "A0": [1],
        "tau": [10],
        "w":  [0.2],
        "A0_pr": [0],
        "te": 0,
        "tau_pr": [1.5],
        "w_pr": [1],
        "t_delay": [0],
        "te_pr": 0
    }
]
sc=1


params = []
for p in params_:
    for Ne in p["Ne"]:
        for tmin in p["tmin"]:
            for tmax in p["tmax"]:
                for T in p["T"]:
                    for wd in p["wd"]:
                        for m in p["m"]:
                            for ef in p["ef"]:
                                for v in p["v"]:
                                    for tau in p["tau"]:
                                        for w in p["w"]:
                                            for tau_pr in p["tau_pr"]:
                                                for w_pr in p["w_pr"]:
                                                    for g in p["g"]:
                                                        for A0 in p["A0"]:
                                                            for A0_pr in p["A0_pr"]:
                                                                for t_delay in p["t_delay"]:
                                                                    if A0_pr == 0 and A0 == 0:
                                                                        pass
                                                                    else:
                                                                        params.append({
                                                                            "Ne": Ne,
                                                                            "tmin": tmin,
                                                                            "tmax": tmax,
                                                                            "Nt": p["Nt"],
                                                                            "T": T,
                                                                            "wd": wd,
                                                                            "s": p["s"],
                                                                            "m": m,
                                                                            "ef": ef,
                                                                            "g": g,
                                                                            "pre_d0": p["pre_d0"],
                                                                            "v": v,
                                                                            "A0": A0,
                                                                            "tau": tau,
                                                                            "w":  w,
                                                                            "te": p["te"],
                                                                            "A0_pr": A0_pr,
                                                                            "tau_pr": tau_pr,
                                                                            "w_pr": w_pr,
                                                                            "t_delay": t_delay,
                                                                            "te_pr": p["te_pr"]
                                                                        })

print(len(params),'parameters generated')
# params = [params[0]]


p = params[0]

Ne = p["Ne"]
tmin= p["tmin"]
tmax= p["tmax"]
Nt=p["Nt"]

t_points = np.linspace(tmin, tmax, Nt)


#superconductor parameters
T = p["T"]
wd = p["wd"]
s = p["s"]
nb = len(s) #nb=number of bands
m= p["m"]
ef= p["ef"]
# gscale=np.array([10])
g = p["g"]
pre_d0 = p["pre_d0"]
v_legget = p["v"]
#pump parameters
A0 = p["A0"]
tau = p["tau"]
w = p["w"]
te = p["te"]
te_pr = p["te_pr"]

# Second pulse
A0_pr = p["A0_pr"]
tau_pr = p["tau_pr"]
w_pr = p["w_pr"]
t_delay = p["t_delay"]

#Constants
hbar=1
kb=1
e_charge=1

# Calculate remaining dependent parameters from these parameters
B = 1/(kb*T)
kf = np.sqrt(2*m*ef)/hbar
vf = kf/m
n = kf**3/(3*np.pi**2)
N0 = m*kf/(2*np.pi**2)

def pfft(t,data):
    N_t = len(t)
    y= data
    yf = scipy.fft.fft(y)
    xf = fftfreq(N_t, t[1]-t[0])
    xf = fftshift(xf)
    yplot = fftshift(yf)
    return np.array([xf*2*np.pi, 1.0/N_t*yplot])

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

def nm(x):
    return x / np.max(np.abs(x))

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

# if U is set, calculate d0 like this:
def d0_integrand(x, d):
    # This is an auxiliary function used in find_d0 to calculate an integral
    return 0.5*1/np.sqrt(x**2+d**2)*np.tanh(B/2*np.sqrt(x**2+d**2))

def find_d0(UN0):
    # this function finds the initial gap(s) given U*N(0). Works in the single or multiband case
    if nb == 2:
        d = np.array([1, 1])
        integral = np.zeros(2)
        for j in [0, 1]:
            integral[j] = integrate.quad(d0_integrand, -wd, wd, (d[j],))[0]
        d_new = np.sum(UN0*d*integral, axis=1)

        while (np.linalg.norm(d - d_new) > 1e-15):
            d = d_new
            integral = np.zeros(2)
            for j in [0, 1]:
                integral[j] = integrate.quad(d0_integrand, -wd, wd, (d[j],))[0]
            d_new = np.sum(UN0*d*integral, axis=1)
        return d_new
    elif nb == 1:
        UN0 = U[0]*N0
        d = 1
        d_new = UN0*d*integrate.quad(d0_integrand, -wd, wd, (d,))[0]
        while (np.linalg.norm(d - d_new) > 1e-15):
            d = d_new
            d_new = UN0*d*integrate.quad(d0_integrand, -wd, wd, (d,))[0]
        return d_new

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


def find_U(deq,v):
    #calculation of remaining U parameters:
    if nb==2:
        U = np.array([[1,1],
                [1,1]])

        I = np.zeros(2)
        for j in [0, 1]:
            I[j] = integrate.quad(d0_integrand, -wd, wd, (deq[j],))[0]

        U11 = (deq[0]-U[0,1]*N0[1]*deq[1]*I[1])/(N0[0]*I[0]*deq[0])
        U22 = (deq[1]-U[1,0]*N0[0]*deq[0]*I[0])/(N0[1]*I[1]*deq[1])
        U12 = v*U11
        U_new=np.array([[U11,U12],
                [U12,U22]])

        while np.linalg.norm(U-U_new)>1e-10:
            U=U_new
            d1=U[0,0]*N0[0]*deq[0]*I[0]+U[0,1]*N0[1]*deq[1]*I[1]
            d2=U[1,0]*N0[0]*deq[0]*I[0]+U[1,1]*N0[1]*deq[1]*I[1]

            U11 = (deq[0]-U[0,1]*N0[1]*deq[1]*I[1])/(N0[0]*I[0]*deq[0])
            U22 = (deq[1]-U[1,0]*N0[0]*deq[0]*I[0])/(N0[1]*I[1]*deq[1])
            U12 = v*U[0,0]
            U_new=np.array([[U11,U12],
                    [U12,U22]])
        return U_new
    else:
        return("Leggett mode only for multiband")

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



B = 1/(kb*0.000001)
ep = np.linspace(-wd, wd, Ne)
U = find_U2(pre_d0,v_legget)
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


ne_ = Ne # number of equations per band
ne = nb * ne_ # number of equations

ax = np.newaxis

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

# indices are:  [band,k1,k2,k3]
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



def integ(x, axis):
    """ Integrate the function 'x' over the axis 'axis'. The integration can be performed over one or two dimensions """
    if hasattr(axis, "__len__"):
        if len(axis) == 2:
            return integrate.simps(integrate.simps(x, dx=de, axis=axis[1]), dx=de, axis=axis[0])
        elif len(axis) == 3:
            return integrate.simps(integrate.simps(integrate.simps(x, dx=de, axis=axis[2]), dx=de, axis=axis[1]), dx=de, axis=axis[0])
    else:
        return integrate.simps(x, dx=de, axis=axis)
    """
    dx = de
    if hasattr(axis, "__len__"):
        dx = de**(len(axis))
    return np.sum(x, axis=axis) * dx
    """

def A(t):
    # return A_pump(t) + A_probe(t)
    """ Returns the vector potential at time t """
    return A0*np.exp(-(t-te)**2/(2*tau**2))*np.cos(w*t) \
        +  A0_pr*np.exp(-(t-te-t_delay)**2/(2*tau_pr**2))*np.cos(w_pr*(t-t_delay))

#%% optical conductivity first order

eta = 0.1
w = 4
w = w + 1j*eta

ek,ek2,eek,eek2,d,W12,nfeek,nfeek2,nfeekm,nfeek2m,fk,fk2=genE(2)
# finite temp
x00_ = (nfeek2*(d**2 + eek2**2 + ek*ek2 - eek2*w))/(eek2*(-eek**2 + (eek2 - w)**2)) - (nfeekm*(d**2 + eek**2 + ek*ek2 - eek*w))/(eek*(eek**2 - eek2**2 - 2*eek*w + w**2)) + (nfeek*(d**2 + ek*ek2 + eek*(eek + w)))/(eek*(eek - eek2 + w)*(eek + eek2 + w)) - (nfeek2m*(d**2 + ek*ek2 + eek2*(eek2 + w)))/(eek2*(-eek**2 + (eek2 + w)**2))

plt.figure('x00', figsize=(3.2,2.8))
plt.clf()
data = -x00_[1].imag
m = np.max(data)
plt.pcolormesh(e1_,e1_,data, cmap='OrRd')#,vmin=-m, vmax=m)
plt.axis('equal')
plt.colorbar()
plt.xlabel('$\epsilon$')
plt.ylabel('$\epsilon\'$')
plt.text(-4,4,'$\omega=4$')
plt.tight_layout()
plt.savefig('x00.png', dpi=800)

#%%
eta = 0.01
ws = np.linspace(0,4,50)
w = ws + 1j*eta

ek,ek2,ek3,eek,eek2,eek3,d,W12,W13,W23,nfeek,nfeek2,nfeek3,nfeekm,nfeek2m,nfeek3m,fk,fk2,fk3=genE(3)
# finite temp
x00_ = (nfeek2*(d**2 + eek2**2 + ek*ek2 - eek2*w))/(eek2*(-eek**2 + (eek2 - w)**2)) - (nfeekm*(d**2 + eek**2 + ek*ek2 - eek*w))/(eek*(eek**2 - eek2**2 - 2*eek*w + w**2)) + (nfeek*(d**2 + ek*ek2 + eek*(eek + w)))/(eek*(eek - eek2 + w)*(eek + eek2 + w)) - (nfeek2m*(d**2 + ek*ek2 + eek2*(eek2 + w)))/(eek2*(-eek**2 + (eek2 + w)**2))
integral = integ(x00_[1], axis=(0,1))

#%%
plt.figure('22')
plt.clf()
plt.plot(ws, -integral.imag)
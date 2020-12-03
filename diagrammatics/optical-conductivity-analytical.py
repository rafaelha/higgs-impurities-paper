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
        "Ne": [4000],
        "tmin": [-80],
        "tmax": [300],
        # "tmax": [450],
        "Nt": 450,
        "T": [4/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10,5])],
        # "g": [ r['g'] ],
        "pre_d0": np.array([0.3,0.7]),
        # "pre_d0": np.array([0.29817841,0.70076507]),
        "v": [0.8],#np.linspace(0.0002,1,40),
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

# eta = 0.005
# ws1 = np.linspace(0,1.4*4,20)
# ws2 = np.logspace(-1.5,0.5,30)
# ws = np.sort(np.concatenate([ws1,ws2]))

# w = ws + 1j*eta
# # expression = -((d**4 + eek2**4 + ek**2*ek2*ek3 - 4*eek2**3*w + ek2*(4*ek + ek3)*w**2 - d**2*(-6*eek2**2 + ek**2 + ek2*ek3 - 2*ek*(ek2 + ek3) + 12*eek2*w - 5*w**2) - 2*eek2*w*(ek*(ek + 3*ek2) + (ek + ek2)*ek3 + w**2) + eek2**2*(ek**2 + ek2*ek3 + 2*ek*(ek2 + ek3) + 5*w**2))/(eek2*(eek2 - eek3 - 2*w)*(eek2 + eek3 - 2*w)*(eek + eek2 - w)**2*(eek - eek2 + w)**2)) - (d**4 + eek3**4 + ek**2*ek2*ek3 + 4*eek3**3*w + (4*ek + ek2)*ek3*w**2 + 2*eek3*w*(ek*(ek + ek2) + (3*ek + ek2)*ek3 + w**2) + eek3**2*(ek**2 + ek2*ek3 + 2*ek*(ek2 + ek3) + 5*w**2) + d**2*(6*eek3**2 - ek**2 - ek2*ek3 + 2*ek*(ek2 + ek3) + 12*eek3*w + 5*w**2))/(eek3*(eek**2 - (eek3 + w)**2)**2*(-eek2**2 + (eek3 + 2*w)**2)) + (eek**8 + 4*eek**5*(-eek2**2 + eek3**2 + 2*ek*(-ek2 + ek3))*w + eek**6*(eek2**2 + eek3**2 + 3*ek**2 + 3*ek2*ek3 + 6*ek*(ek2 + ek3) - w**2) - 4*eek*(eek2 - eek3)*(eek2 + eek3)*ek**2*w*(-(ek2*ek3) + w**2) + ek**2*(eek2 - w)*(eek2 + w)*(-eek3**2 + w**2)*(-(ek2*ek3) + w**2) + 4*eek**3*ek*(ek2 - ek3)*w*(eek2**2 + eek3**2 + 2*w**2) - eek**4*(-5*ek**2*ek2*ek3 + (7*ek**2 + 2*ek2*ek3 + 4*ek*(ek2 + ek3))*w**2 + w**4 + eek3**2*(ek**2 + ek2*ek3 + 2*ek*(ek2 + ek3) - 4*w**2) + eek2**2*(3*eek3**2 + ek**2 + ek2*ek3 + 2*ek*(ek2 + ek3) - 4*w**2)) + d**4*(5*eek**4 + 4*eek*(eek2 - eek3)*(eek2 + eek3)*w + (eek2 - w)*(eek3 - w)*(eek2 + w)*(eek3 + w) - 3*eek**2*(eek2**2 + eek3**2 + 2*w**2)) - eek**2*(6*ek**2*ek2*ek3*w**2 + (-5*ek**2 + ek2*ek3 + 2*ek*(ek2 + ek3))*w**4 - w**6 + eek3**2*(3*ek**2*ek2*ek3 - (4*ek**2 + 6*ek*ek2 - 2*ek*ek3 + ek2*ek3)*w**2 + w**4) + eek2**2*(3*ek**2*ek2*ek3 - (4*ek**2 - 2*ek*ek2 + 6*ek*ek3 + ek2*ek3)*w**2 + w**4 + eek3**2*(ek**2 + ek2*ek3 + 2*ek*(ek2 + ek3) - w**2))) + d**2*(18*eek**6 - 4*eek*(eek2 - eek3)*(eek2 + eek3)*w*(ek**2 + ek2*ek3 - 2*ek*(ek2 + ek3) + w**2) + (eek2 - w)*(eek2 + w)*(-eek3**2 + w**2)*(ek**2 + ek2*ek3 - 2*ek*(ek2 + ek3) + w**2) - eek**4*(6*eek2**2 + 6*eek3**2 + 5*(ek**2 + ek2*ek3 - 2*ek*(ek2 + ek3)) + 17*w**2) + 3*eek**2*(2*(ek**2 + ek2*ek3 - 2*ek*(ek2 + ek3))*w**2 + eek3**2*(ek**2 + ek2*ek3 - 2*ek*(ek2 + ek3) + 3*w**2) + eek2**2*(-2*eek3**2 + ek**2 + ek2*ek3 - 2*ek*(ek2 + ek3) + 3*w**2))))/(2.*eek**3*(eek + eek3 - w)**2*(eek - eek2 + w)**2*(eek + eek2 + w)**2*(-eek + eek3 + w)**2)
# # integral = N0**3 * integ(W12 * W13 * expression, axis=(1,2,3))
# # expression = -((d**2 + eek**2 + ek*ek2 - eek*w)/(eek*(eek**2 - eek2**2 - 2*eek*w + w**2))) - (d**2 + eek2**2 + ek*ek2 + eek2*w)/(eek2*(-eek**2 + eek2**2 + 2*eek2*w + w**2))
# # expression = (nfeek2*(d**2 + eek2**2 + ek*ek2 - eek2*w))/(eek2*(-eek**2 + (eek2 - w)**2)) - (nfeekm*(d**2 + eek**2 + ek*ek2 - eek*w))/(eek*(eek**2 - eek2**2 - 2*eek*w + w**2)) + (nfeek*(d**2 + ek*ek2 + eek*(eek + w)))/(eek*(eek - eek2 + w)*(eek + eek2 + w)) - (nfeek2m*(d**2 + ek*ek2 + eek2*(eek2 + w)))/(eek2*(-eek**2 + (eek2 + w)**2))

# ek,ek2,ek3,eek,eek2,eek3,d,W12,W13,W23,nfeek,nfeek2,nfeek3,nfeekm,nfeek2m,nfeek3m,fk,fk2,fk3=genE(3)
# # zero temp
# # x00_ = (1-(ek*ek2+d**2)/(eek*eek2)) * (eek+eek2)/(w**2-(eek+eek2)**2)
# # finite temp
# x00_ = (nfeek2*(d**2 + eek2**2 + ek*ek2 - eek2*w))/(eek2*(-eek**2 + (eek2 - w)**2)) - (nfeekm*(d**2 + eek**2 + ek*ek2 - eek*w))/(eek*(eek**2 - eek2**2 - 2*eek*w + w**2)) + (nfeek*(d**2 + ek*ek2 + eek*(eek + w)))/(eek*(eek - eek2 + w)*(eek + eek2 + w)) - (nfeek2m*(d**2 + ek*ek2 + eek2*(eek2 + w)))/(eek2*(-eek**2 + (eek2 + w)**2))
# x00 = - N0[:,ax]**2 * integ(W12 * x00_, axis=(1,2)) #sign!!!
# jp_1 = vf[:,ax]**2/3/N0[:,ax] * x00


# ek,ek2,eek,eek2,d,W12,nfeek,nfeek2,nfeekm,nfeek2m,fk,fk2=genE(2)
# eep = (ek - ek2) * np.ones((nb,1,1))
# for i in np.arange(nb):
#     np.fill_diagonal(eep[i], 1)
# jd_1 = (n/m) * integ( (fk-fk2)  / eep * W12, axis=(1,2))
# jd_1 = jd_1[:,ax]

# j_1 = jd_1+jp_1
# cond = np.sum(j_1, axis=0) / (1j * ws)
# sigma = cond*u_conductivity

# plt.figure('cond-imp-real', figsize=(3.2,2.8))
# plt.clf()
# plt.subplot(121)
# # plt.plot(wg*u_w,np.abs(c.real), label=f'$\gamma/2\Delta={str(np.round(g[0]/2/gap,1))}$')
# # g = r['g']
# # plt.title(f'$\gamma_1={g[0]}, \gamma_2={g[1]}$')
# plt.xlim((0,6))
# plt.axvline(2*d_eq0[0]*u_w, c='gray', lw=0.5)
# plt.axvline(2*d_eq0[1]*u_w, c='gray', lw=0.5)

# plt.plot(ws*u_w, sigma.real, '.')
# plt.ylabel('$\sigma\,\'$ ($\Omega^{-1}$cm$^{-1}$)')
# plt.xlabel('Frequency (THz)')
# plt.ticklabel_format(axis="y", style="sci", scilimits=(2,4))
# plt.ylim((0,np.max(np.nan_to_num(np.abs(sigma.real))[ws*u_w<6])*1.2))
# plt.tight_layout()

# plt.subplot(122)
# # plt.loglog(wg*u_w,c.imag, '.-', label=f'$\gamma/2\Delta={str(np.round(g[0]/2/gap,1))}$')
# plt.xlim((0,6))
# plt.axvline(2*d_eq0[0]*u_w, c='gray', lw=0.5)
# plt.axvline(2*d_eq0[1]*u_w, c='gray', lw=0.5)
# plt.plot(ws*u_w, sigma.imag, '.')
# plt.ylabel('$\sigma\,\'\'$ ($\Omega^{-1}$cm$^{-1}$)')
# plt.xlabel('Frequency (THz)')
# # plt.ticklabel_format(axis="y", style="sci", scilimits=(2,4))
# # plt.ylim((0,0.5e5))
# plt.xlim((-2,6))
# plt.tight_layout()

#%% Leggett current third order

eta = 0.01
ws = np.linspace(0,1,1000)

w = 2*ws[ax,ax,:] + 1j*eta
w2 = 2*ws+ 1j*eta

ek,ek2,eek,eek2,d,W12,nfeek,nfeek2,nfeekm,nfeek2m,fk,fk2 = genE(2)

x33_ = (-2*(d**2 + eek**2 - ek**2))/(4*eek**3 - eek*w**2)
x33 = N0[:,ax] * integ(x33_, axis=1)


kappa = 8*d_eq0[0]*d_eq0[1]*U[0,1]/np.linalg.det(U)
si = s[:,ax]
mi = m[:,ax]

det = (-x33[0]-kappa/w2**2)*(-x33[1]-kappa/w2**2) - kappa**2/w2**4
Linv = 4 / det * np.array([[-x33[1]-kappa/w2**2,  -kappa/w2**2], \
                                 [-kappa/w2**2,         -x33[0]-kappa/w2**2]])

jL = 4*2**(-6) * np.einsum('iw,ijw,jw->w',x33*si/mi,Linv,x33*si/mi)
jQPdia = (si/2/mi)**2 * x33
plt.figure('jL')
plt.clf()
plt.plot(ws,np.abs(jL))
plt.plot(ws,np.abs(np.sum(jQPdia,0)))
plt.axvline(d_eq0[0], c='gray', lw=0.5)
plt.axvline(d_eq0[1], c='gray', lw=0.5)

plt.figure('jL-diff')
plt.clf()
# plt.plot(ws,np.abs(np.sum(jQPdia,0))/np.abs(jL))
plt.plot(ws,np.abs(np.sum(jQPdia,0)+jL))
plt.axvline(d_eq0[0], c='gray', lw=0.5)
plt.axvline(d_eq0[1], c='gray', lw=0.5)
# plt.ylim((0,5))

plt.figure('det')
plt.clf()
plt.plot(ws, np.abs(1/det))
# plt.plot(ws, x33[0])
# plt.plot(ws, x33[1])
plt.axvline(d_eq0[0], c='gray', lw=0.5)
plt.axvline(d_eq0[1], c='gray', lw=0.5)

#%% QP diamagnetic third order

eta = 0.01
ws = np.linspace(0,1,1000)

w = 2*ws[ax,ax,:] + 1j*eta
w2 = 2*ws+ 1j*eta

ek,ek2,eek,eek2,d,W12,nfeek,nfeek2,nfeekm,nfeek2m,fk,fk2 = genE(2)

x33_ = (-2*(d**2 + eek**2 - ek**2))/(4*eek**3 - eek*w**2)
x33 = N0[:,ax] * integ(x33_, axis=1)

si = s[:,ax]
mi = m[:,ax]


jQPdia = (si/2/mi)**2 * x33
plt.figure('jQPdia')
plt.clf()
plt.plot(ws,np.abs(jQPdia).T)
plt.axvline(d_eq0[0], c='gray', lw=0.5)
plt.axvline(d_eq0[1], c='gray', lw=0.5)


#%%

# dim = 2
# ws = np.linspace(0,2,200)

# eta = 0.002
# w = ws[ax,ax,:] + 1j*eta

# # expression = (nfeek2*(d**2 + eek2**2 + ek*ek2 - eek2*w))/(eek2*(-eek**2 + (eek2 - w)**2)) - (nfeekm*(d**2 + eek**2 + ek*ek2 - eek*w))/(eek*(eek**2 - eek2**2 - 2*eek*w + w**2)) + (nfeek*(d**2 + ek*ek2 + eek*(eek + w)))/(eek*(eek - eek2 + w)*(eek + eek2 + w)) - (nfeek2m*(d**2 + ek*ek2 + eek2*(eek2 + w)))/(eek2*(-eek**2 + (eek2 + w)**2))
# x11_ = (-2*(-d**2 + eek**2 + ek**2))/(4*eek**3 - eek*w**2)
# x11 = integ(x11_, axis=1)
# dU = U[0,0]*U[1,1]-U[0,1]**2
# det = (x11[0]+2*U[1,1]/dU)*(x11[1]+2*U[0,0]/dU) - (2*U[0,1]/dU)**2



# plt.figure('chi')
# plt.clf()
# chi = np.stack(chi)

# # plt.plot(ws, np.real(x11.T) + np.array([2/U[0,0], 2/U[1,1]]))
# # plt.plot(ws,np.abs(det.real))
# # plt.plot(ws,np.abs(det.imag))
# plt.plot(ws,1/np.abs(det))
# plt.axvline(2*d_eq0[0], c='gray', lw=0.5)
# plt.axvline(2*d_eq0[1], c='gray', lw=0.5)
# # plt.plot(ws,chi.real)
# # plt.plot(ws,chi.imag)


# %%

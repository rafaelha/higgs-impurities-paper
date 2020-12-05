#%%
from operator import mul
import numpy as np
from numpy import sqrt
import matplotlib as mpl
mpl.use('Agg')
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
import time
from multiprocessing import Pool

code_version = {
    "1.0": "first version",
}

if len(sys.argv) != 3:
    job_ID = -1
    task_ID = job_ID
    task_count = 0
else:
    job_ID = int(sys.argv[1])
    task_ID = int(sys.argv[1])
    task_count = int(sys.argv[2])+1

u_t = 6.58285E-2
u_e = 10
u_conductivity = 881.553 #Ohm^-1 cm-1
meV_to_THz = 0.2417990504024
u_w = u_e*meV_to_THz

u_temp = 116.032

parallel = False
filtename = 'results-j3'
params_ = [
    {
        "Ne": [400],
        "w":  0.3/u_w,
        "T": np.arange(0,60,100)/u_temp,
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10,5])],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0.5],#np.linspace(0.0002,1,40),
        "eta":  [0.01],
    }
]


params = []
for p in params_:
    for Ne in p["Ne"]:
        for T in p["T"]:
            for wd in p["wd"]:
                for m in p["m"]:
                    for ef in p["ef"]:
                        for v in p["v"]:
                            for w in p["w"]:
                                for eta in p["eta"]:
                                    for g in p["g"]:
                                        params.append({
                                                        "Ne": Ne,
                                                        "T": T,
                                                        "wd": wd,
                                                        "s": p["s"],
                                                        "m": m,
                                                        "ef": ef,
                                                        "g": g,
                                                        "pre_d0": p["pre_d0"],
                                                        "v": v,
                                                        "w":  w,
                                                        "eta": eta
                                                    })

print(len(params),'parameters generated')

if task_ID != -1:
    params = params[task_ID::task_count]
else:
    params = [ params[0] ]

for p in params:
    Ne = p["Ne"]

    start = time.time()

    #superconductor parameters
    T = p["T"]
    wd = p["wd"]
    s = p["s"]
    nb = len(s) #nb=number of bands
    m= p["m"]
    ef= p["ef"]
    g = p["g"]
    pre_d0 = p["pre_d0"]
    v_legget = p["v"]
    #pump parameters
    w_point = p["w"]
    eta = p["eta"]

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

    zerotemp = False
    if T*u_temp < 0.01:
        zerotemp = True

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


#%% Leggett current third order

    w = 2*w_point + 1j*eta

    ek,eek,d,nfeek,nfeekm = genE(1)

    x33_ = (-2*(d**2 + eek**2 - ek**2))/(4*eek**3 - eek*w**2)
    x33 = N0 * integ(x33_, axis=1)

    kappa = 8*d_eq0[0]*d_eq0[1]*U[0,1]/np.linalg.det(U)

    det = (-x33[0]-kappa/w**2)*(-x33[1]-kappa/w**2) - kappa**2/w**4
    Linv = 4 / det * np.array([[-x33[1]-kappa/w**2,  -kappa/w**2], \
                                    [-kappa/w**2,         -x33[0]-kappa/w**2]])

    jphase = 4 * 1/2 * 2**(-4) * np.einsum('i,ij,j->',x33*s/m,Linv,x33*s/m) # 4 legs to do functional derivation
    jQPdia = 4* 1/2* (s/2/m)**2 * x33 #paramagnetic current at third order
    jL = np.sum(jQPdia,0)+jphase

    jL2 = -4 * 1/2*1/4* kappa * (s/m-s/m)**2 / (-w**2 - kappa*(x33[0]+x33[1])/(x33[0]*x33[1]))

    d_theta = 2 * 2**(-3) * np.einsum('ij,j->i',Linv,x33*s/m)
    d_phi = d_theta[0]-d_theta[1]

#%% Higgs current third order
    # Higgs propagator
    w = w_point + 1j*eta

    ek,eek,d,nfeek,nfeekm = genE(1)
    # Tr[g[ek, eek, z].s1.g[ek, eek, z + 2 w].s1]
    x11_ = -((-d**2 + eek**2 + ek**2)/(2*eek**3 - 2*eek*w**2))
    x11 = N0*integ(x11_, axis=1)

    dU = U[0,0]*U[1,1]-U[0,1]**2
    det = (x11[0]+2*U[1,1]/dU)*(x11[1]+2*U[0,0]/dU) - (2*U[0,1]/dU)**2
    Hinv = 1/det * np.array([[x11[1]+2*U[0,0]/dU,    2*U[0,1]/dU],\
                            [2*U[0,1]/dU,            x11[0]+2*U[1,1]/dU  ]])

    # susceptibilities
    ek,ek2,eek,eek2,d,W12,nfeek,nfeek2,nfeekm,nfeek2m,fk,fk2 = genE(2)
    pre_W = vf**2/3/N0

    jH = []

    # Tr[g[ek, eek, z].s1.g[ek, eek, z - 2 w].g[ek2, eek2, z + w]]
    x100l_= (d*((eek + eek2)**2*(3*eek**2*eek2 - d**2*(2*eek + eek2) + 2*eek*ek*(ek - 2*ek2) + eek2*ek*(ek - 2*ek2)) + (-6*eek**3 + 3*eek**2*eek2 - 2*eek2**3 + d**2*(2*eek + 5*eek2) - 5*eek2*ek*(ek - 2*ek2) - 2*eek*(eek2**2 + ek**2 - 2*ek*ek2))*w**2 + 6*(eek - eek2)*w**4))/(2.*eek*eek2*(eek + eek2 - 3*w)*(eek - w)*(eek + eek2 - w)*(eek + w)*(eek + eek2 + w)*(eek + eek2 + 3*w))
    x100l = pre_W * N0**2 * integ(W12*x100l_, axis=(1,2))

    # Tr[g[ek, eek, z].s1.g[ek, eek, z + 2 w].g[ek2, eek2, z + w]]
    x100r_= (d*(3*eek**2*eek2 - d**2*(2*eek + eek2) + eek2*(ek**2 - 2*ek*ek2 - 2*w**2) + 2*eek*(ek**2 - 2*ek*ek2 + w**2)))/(2.*eek*eek2*(eek - w)*(eek + eek2 - w)*(eek + w)*(eek + eek2 + w))
    x100r = pre_W * N0**2 * integ(W12*x100r_, axis=(1,2))

    jH = 4*1/2* np.einsum('i,ij,j->',x100l,Hinv,x100r)



#%% QP current third order (parallelized code)
    w = w_point + 1j*eta

    # susceptibilities
    ek,ek2,eek,eek2,d,W12,nfeek,nfeek2,nfeekm,nfeek2m,fk,fk2 = genE(2)
    pre_W = vf**2/3/N0

    def multiprocess(ik):
        ek3 = ek[:,ik,:,ax]
        eek3 = eek[:,ik,:,ax]
        W13 = W12[:,:,ik]
        # Tr[g[ek, eek, z].g[ek3, eek3, z - w].g[ek, eek, z - 2 w].g[ek2, eek2, z + w]]
        x3333_ = -(d**4 + 6*d**2*eek**2 + eek**4 - d**2*ek**2 + eek**2*ek**2 + 2*d**2*ek*ek2 + 2*eek**2*ek*ek2 + 2*d**2*ek*ek3 + 2*eek**2*ek*ek3 - d**2*ek2*ek3 + eek**2*ek2*ek3 + ek**2*ek2*ek3 - 18*d**2*eek*w - 6*eek**3*w - 4*eek*ek**2*w - 4*eek*ek*ek2*w - 8*eek*ek*ek3*w - 2*eek*ek2*ek3*w + 11*d**2*w**2 + 11*eek**2*w**2 + 3*ek**2*w**2 + 2*ek*ek2*w**2 + 6*ek*ek3*w**2 - 6*eek*w**3)/(2.*eek*w*(-2*eek + 2*w)*(eek**2 - eek3**2 - 2*eek*w + w**2)*(eek**2 - eek2**2 - 6*eek*w + 9*w**2)) + (d**4 + 6*d**2*eek**2 + eek**4 - d**2*ek**2 + eek**2*ek**2 + 2*d**2*ek*ek2 + 2*eek**2*ek*ek2 + 2*d**2*ek*ek3 + 2*eek**2*ek*ek3 - d**2*ek2*ek3 + eek**2*ek2*ek3 + ek**2*ek2*ek3 + 6*d**2*eek*w + 2*eek**3*w + 4*eek*ek*ek2*w + 2*eek*ek2*ek3*w - d**2*w**2 - eek**2*w**2 - ek**2*w**2 + 2*ek*ek2*w**2 - 2*ek*ek3*w**2 - 2*eek*w**3)/(2.*eek*(-2*eek - 2*w)*w*(eek**2 - eek2**2 - 2*eek*w + w**2)*(eek**2 - eek3**2 + 2*eek*w + w**2)) - (d**4 + 6*d**2*eek2**2 + eek2**4 - d**2*ek**2 + eek2**2*ek**2 + 2*d**2*ek*ek2 + 2*eek2**2*ek*ek2 + 2*d**2*ek*ek3 + 2*eek2**2*ek*ek3 - d**2*ek2*ek3 + eek2**2*ek2*ek3 + ek**2*ek2*ek3 + 18*d**2*eek2*w + 6*eek2**3*w + 2*eek2*ek**2*w + 8*eek2*ek*ek2*w + 4*eek2*ek*ek3*w + 4*eek2*ek2*ek3*w + 11*d**2*w**2 + 11*eek2**2*w**2 + 8*ek*ek2*w**2 + 3*ek2*ek3*w**2 + 6*eek2*w**3)/(eek2*(-eek**2 + eek2**2 + 2*eek2*w + w**2)*(eek2**2 - eek3**2 + 4*eek2*w + 4*w**2)*(-eek**2 + eek2**2 + 6*eek2*w + 9*w**2)) - (d**4 + 6*d**2*eek3**2 + eek3**4 - d**2*ek**2 + eek3**2*ek**2 + 2*d**2*ek*ek2 + 2*eek3**2*ek*ek2 + 2*d**2*ek*ek3 + 2*eek3**2*ek*ek3 - d**2*ek2*ek3 + eek3**2*ek2*ek3 + ek**2*ek2*ek3 - 6*d**2*eek3*w - 2*eek3**3*w - 2*eek3*ek**2*w - 4*eek3*ek*ek3*w - d**2*w**2 - eek3**2*w**2 - ek2*ek3*w**2 + 2*eek3*w**3)/(eek3*(-eek**2 + eek3**2 - 2*eek3*w + w**2)*(-eek**2 + eek3**2 + 2*eek3*w + w**2)*(-eek2**2 + eek3**2 - 4*eek3*w + 4*w**2))
        x3333 = pre_W**2 * N0**2 *integ(W13 * integ(W12 * x3333_, axis=2), axis=1)
        return x3333

    jQP = []
    durations = []

    if parallel:
        pool = Pool()
        multiresult = np.stack( pool.map(multiprocess, np.arange(Ne)) ).T
        pool.close()
    else:
        multiresult = np.stack([multiprocess(i) for i in np.arange(Ne)]).T
    jQP = 1/4 * 4 * N0 * integ(multiresult, axis=1)


#%% Save everything
    end = time.time()
    duration = (end - start)/60
    print(f' finished in {duration}s')

    res = {'Ne': Ne,
            'wd': wd,
            's': s,
            'm': m,
            'ef': ef,
            'g': g,
            'U': U,
            'd_eq': d_eq,
            'nb': nb,

            'T': T,
            'w': w,
            'jQP': jQP,
            'jL': jL,
            'jL2': jL2,
            'jH': jH,

            'duration': duration,
            'hbar': hbar,
            'kb': kb,
            'e_charge': e_charge,
            'job_ID': job_ID,
            'task_ID': task_ID,
            'task_count': task_count,
            'version': code_version}


    # f1 = open(f'{job_ID}_{task_ID}.pickle', 'ab')
    f1 = open(f'result-j3.pickle', 'ab')
    pickle.dump(res, f1)
    f1.close()
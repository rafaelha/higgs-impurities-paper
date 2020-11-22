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

u_temp = 116.032
params_ = [
    {
        "Ne": [1000],
        "tmin": [-80],
        "tmax": [300],
        # "tmax": [450],
        "Nt": 450,
        "T": [0.001/u_temp],#,0.5,0.54,0.56],
        "wd":  [5],
        "s": np.array([1,-1]),
        "m": [ np.array([0.85, 1.38]) ],
        "ef": [ np.array([290, 70]) ],
        "g": [ np.array([10,5])],
        "pre_d0": np.array([0.3,0.7]),
        "v": [0.1],#np.linspace(0.0002,1,40),
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


for p in params:
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

    # if gscale.any() != None:
        # g = gscale*(2*d_eq0)

    ne_ = Ne # number of equations per band
    ne = nb * ne_ # number of equations

    # Variables used by ode solvers. They are computed once on script execution and then used many times

    # We choose the following naming convention:

    # variable name ending on 1_ indicates an 1D array of dimention [Ne] corresponding to energies epsilon in [-wd,wd]
    # name ending on 1 indicates an array of dimension [nb,Ne] that also includes the band degree of freedom
    # no ending indicates array of dimension [nb,Ne,Ne] where first index is for band, second is for epsilon,
    # and the last index is epsilon'

    # we will also make use of python broadcasting. If a variable does not depend on an index, we may insert a 
    # dimension with only one element using ax. I.e. the 3D array e only depends on the epsilon index, but to make it 
    # 3D we insert placeholder axes in the following way e = e1_[ax,:,ax]

    ax = np.newaxis

    d_eq1 = d_eq0[:,ax]
    d_eq = d_eq0[:,ax, ax]
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

    u1 = np.sqrt(0.5 * (1 + e1/E1))
    v1 = np.sqrt(0.5 * (1 - e1/E1))
    u = np.sqrt(0.5 * (1 + e/E))
    v = np.sqrt(0.5 * (1 - e/E))
    up = np.sqrt(0.5 * (1 + ep/Ep))
    vp = np.sqrt(0.5 * (1 - ep/Ep))

    l = u*up + v*vp
    p = v*up - u*vp

    W = 1/np.pi*g[:,ax,ax]/((e-ep)**2+g[:,ax,ax]**2)

    f1 = 1.0/(np.exp(B*E1) + 1)
    f = 1.0/(np.exp(B*E) + 1)
    fp = 1.0/(np.exp(B*Ep) + 1)

    ff1_ = 1.0/(np.exp(B*e1_) + 1)
    ff1 = 1.0/(np.exp(B*e1) + 1)
    ff = 1.0/(np.exp(B*e) + 1)
    ffp = 1.0/(np.exp(B*ep) + 1)

    def integ(x, axis):
        """ Integrate the function 'x' over the axis 'axis'. The integration can be performed over one or two dimensions """
        if hasattr(axis, "__len__"):
            return integrate.simps(integrate.simps(x, dx=de, axis=axis[1]), dx=de, axis=axis[0])
        else:
            return integrate.simps(x, dx=de, axis=axis)
        """
        dx = de
        if hasattr(axis, "__len__"):
            dx = de**(len(axis))
        return np.sum(x, axis=axis) * dx
        """
    def diff(x, y):
        x0 = x[:-2]
        x1 = x[1:-1]
        x2 = x[2:]
        y0 = y[:-2]
        y1 = y[1:-1]
        y2 = y[2:]
        f = (x2 - x1)/(x2 - x0)
        der = (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0)
        return np.concatenate([[der[0]],der,[der[-1]]])

    def A(t):
        # return A_pump(t) + A_probe(t)
        """ Returns the vector potential at time t """
        return A0*np.exp(-(t-te)**2/(2*tau**2))*np.cos(w*t) \
            +  A0_pr*np.exp(-(t-te-t_delay)**2/(2*tau_pr**2))*np.cos(w_pr*(t-t_delay))

    plotA(t_points,A(t_points))
    plt.pause(0.01)

#%%
    def sprime(t, s):
        """ function given to the integrator used to calculate the time evolution of the state vector in the second order calculation """
        ds = np.copy(s).reshape(nb, ne_)
        # unpack and reshape all variables from s
        r21e = ds

        # symmetrize
        # r21e = (r21e + np.flip(r21e, axis=1))/2

        # self-consitent recalculation of the gap
        d = 1j * U @ ( N0*integ(r21e.imag, axis=1) )

        # this is for python broadcasting convenience
        d1 = d[:,ax]
        d2 = d[:,ax, ax]

        # second order
        d_r21e = -2*1j*E1 * r21e - 1j*(1-2*f1)/E1 * e_charge**2 * A(t)**2 / 2 * d_eq1 * s1/m1 - (1-2*f1)*d1.imag*sc

        # reshape and repack into ds
        ds = d_r21e

        return ds.reshape((ne,))

    start = time.time()

    # initial conditions - everything is zero since we start in equilibrium
    s0 = np.zeros(ne, dtype=complex)

    t = np.linspace(tmin,tmax,Nt)

    JP1 = []
    JD1 = []
    D2 = []
    EFIELD = []
    JP3 = []
    JD3 = []
    R21E = []
    R21O = []

    # the built in integrator solves for the r values numerically:
    sols = integrate.solve_ivp( sprime, (tmin, tmax), s0, t_eval=t)
    s0 = np.copy(sols.y[:,-1])

    # extracting the solutions from the solver output:
    Y = sols.y.reshape(nb, ne_, len(sols.t)).swapaxes(0,2).swapaxes(1,2)
    t = sols.t
    r21e = Y[:, :, :]

    efield = -diff(t, A(t))

    # compute the second order gap from above solutions for all times
    d_2 = 1j * np.einsum('ij,tj->ti', U, N0*integ(r21e.imag, axis=2))

    tp = t

    end = time.time()
    duration = end - start
    print(f' finished in {duration}s')
#%%

    gg =2*wd* U*N0
    g11 = gg[0,0]
    g22 = gg[1,1]
    g12 = gg[0,1]
    g21 = gg[0,1]
    d1 = pre_d0[0]
    d2 = pre_d0[1]

    mat = np.array([
        [2*d1,0,0,0],
        [0,2*d1-g11,0,-g12],
        [0,0,2*d2,0],
        [0,-g21,0,2*d2-g22]
    ])

    ev, vecs = np.linalg.eigh(mat)



#% plot the data


    plt.figure('delta')
    plt.clf()
    plt.subplot(121)

    tc_ = 0
    tc = tc_

    plt.plot(t,d_2.imag)
    plt.axvspan(tc, np.max(t), facecolor='gray', alpha=0.2)
    sel = (t>tc)
    dp_ = d_2[sel]
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


    plt.figure('Leggett')
    plt.clf()

    plt.subplot(131)
    plt.title(f'v={v_legget}')
    tc = tc_
    dphase = d_2[:,0].imag/d_eq[0,0,0] - d_2[:,1].imag/d_eq[1,0,0]
    plt.axvspan(tc, np.max(t), facecolor='gray', alpha=0.2)
    dp_ = dphase[sel]
    dp_ -= np.mean(dp_, axis=0)
    plt.plot(t,dphase)
    plt.xlabel(f'$t$')
    plt.ylabel('$\\varphi$')

    plt.subplot(132)
    plt.title(f'v={v_legget}')
    tc = tc_
    dphase = d_2[:,0].imag/d_eq[0,0,0] - d_2[:,1].imag/d_eq[1,0,0]
    plt.axvspan(tc, np.max(t), facecolor='gray', alpha=0.2)
    dp_ = dphase[sel]
    dp_ -= np.mean(dp_, axis=0)
    # plt.plot(t,dphase)
    dphase2 = np.copy(dphase)
    dphase2[t<0] = 1
    plt.plot(t,np.log(np.abs(dphase2)))
    plt.xlabel(f'$t$')
    plt.ylabel('log$\\varphi$')

    plt.subplot(133)
    w_, dpw_ = rfft(t_, dp_)
    print(w_[w_>0][np.argmax(np.abs(dpw_[w_>0]))])
    plt.axvline(d_eq[0]*2, c='gray', lw=1)
    plt.axvline(d_eq[1]*2, c='gray', lw=1)
    plt.plot(w_, np.abs(dpw_))
    plt.xlim((0,4*d_eq[1]))
    plt.xlabel(f'$\omega$')
    plt.ylabel('$\\varphi$')

    plt.tight_layout()
    plt.pause(0.01)


    # save as dictionary using pickle
    res = {'Ne': Ne,
            'T': T,
            'wd': wd,
            's': s,
            'm': m,
            'ef': ef,
            #'gscale': gscale,
            'g': g,
            'U': U,
            'v':v_legget,
            'd_eq0_T0': d_eq0_T0,
            'd_eq0': d_eq0,
            'd_eq': d_eq,
            'nb': nb,
            'A0': A0,
            'tau': tau,
            'w': w,
            'te': te,
            'A0_pr': A0_pr,
            'tau_pr': tau_pr,
            'w_pr': w_pr,
            'te_pr': te_pr,
            't_delay': t_delay,
            't': t,
            'A': A(t),
            'd_2': d_2,
            'dphase': dphase,
            'efield': efield,
            'duration': duration,
            'hbar': hbar,
            'kb': kb,
            'e_charge': e_charge,
            'job_ID': job_ID,
            'task_ID': task_ID,
            'task_count': task_count,
            'version': code_version}


    f1 = open(f'{job_ID}_{task_ID}.pickle', 'ab')
    pickle.dump(res, f1)
    f1.close()

#%%
job_ID = 0
task_ID = 0
dphases = []
vs = []

reader = open(f'{job_ID}_{task_ID}.pickle','rb')
try:
    while True:
        a = pickle.load(reader)
        dphases.append(a['dphase'])
        vs.append(a['v'])
except:
    reader.close()

#%%
t = a['t']
temp = a['T']
dphases = np.stack(dphases)
vs = np.array(vs)

tc = 0
sel = (t>tc)
dp_ = dphases[:,sel]
t_ = t[sel]
w_, dpw_ = rfft(t_, dp_.T)
# plt.figure('1')

def nm(x):
    return x/np.max(x,axis=0)

plt.figure('1')
plt.clf()
plt.pcolormesh(vs,w_,nm(np.abs(dpw_)), vmin=0.5, vmax=0.50000000001)
plt.axhline(2*d_eq0[0], c='r')
plt.plot(vs,np.abs(w_[np.argmax(np.abs(dpw_),axis=0)]),'.')
plt.title(f'T={temp*u_temp}K')
# plt.colorbar()

plt.ylim((0,4*d_eq[0]))
plt.xlabel('$v$')
plt.ylabel('$\omega$')
# plt.savefig(f'{job_ID}-tc{tc}.pdf')



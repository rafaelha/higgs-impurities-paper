import numpy as np
from numpy import sqrt
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
from scipy.fftpack import fft, fftfreq, fftshift
import scipy
import time
import pickle
import sys
import os
import gc

code_version = {
    "1.0": "first version",
    "1.1": "pulse shape modified, now parameter te, te_pr"
}
#%%
# ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))       # number of CPUs
job_ID = int(os.environ.get('SLURM_ARRAY_JOB_ID', default=-1))       # job ID
task_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', default=-1)) # task ID
task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', default=1))
task_count = 64

if task_ID != -1:
    exec("from {} import params".format(sys.argv[1]))
    params = params[task_ID::task_count]
else:
    from parameters import params
    params = [ params[0] ]

    params = [
        {
            "Ne": 850,
            "tmin": -2*(2*np.pi),
            "tmax": 30*(2*np.pi),
            "Nt": 3000,
            "T": 0.56, #0.22
            "wd":  10,
            "s": np.array([1]),
            "m": np.array([1.0]),
            "ef": np.array([500]),
            "g": np.array([10]),
            "U": np.array([[0.2082]]),
            "A0": -1,
            "tau": 0.81,
            "w":  0.36,
            "te": -4.21,
            "A0_pr": -0.1,
            "tau_pr": 0.81,
            "w_pr": 0.36,
            "t_delay": 9,
            "te_pr": -4.21
        }
    ]


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
    U =  p["U"]
    #pump parameters
    A0 = p["A0"]
    tau = p["tau"]
    w = p["w"]
    te = p["te"]
    te_pr = p["te_pr"]

    if nb == 2:
        pre_d0  = np.array([0.47921433, 1.06231377])
    elif nb == 1:
        pre_d0  = np.array([0.47921433])
    else:
        pre_d0 = np.ones(nb)

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
    UN0 = U*N0[:, np.newaxis]


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
            while (np.linalg.norm(d - d_new) > 1e-12):
                d = d_new
                d_new = UN0*d*integrate.quad(d0_integrand, -wd, wd, (d,))[0]
            return d_new
    d_eq0 = find_d0(UN0)
    print('gap=',d_eq0)

    # if d0 is set, calculate U like this:
    #U = find_U(pre_d0, 0.05/0.08, integrate.simps(0.5*1/np.sqrt(ep**2+d0.reshape(2,1)**2)*np.tanh(B/2*np.sqrt(ep**2+d0.reshape(2,1)**2)),ep))
    #UN0 = U*N0[:,np.newaxis]

    # if gscale.any() != None:
        # g = gscale*(2*d_eq0)

    ne_ = 3*Ne + 4*Ne**2 # number of equations per band
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
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    # ps = np.loadtxt('..\\material_parameters_Matsunaga\\pulse_shape.csv', delimiter=',')
    # rm = 20
    # t_pulse = running_mean(ps[:,0],rm)

    # b = 0.3
    # smooth = (np.tanh((t_pulse-1.5)/b)+1)/2
    # smooth2 = (-np.tanh((t_pulse-7.5)/b)+1)/2

    # e_pulse = running_mean(ps[:,1],rm)*smooth*smooth2
    # e_pulse /= np.max(np.abs(e_pulse))

    # # 1 time unit = 0.438856 ps
    # t_pulse /= 0.438856

    # a_pulse = -scipy.integrate.cumtrapz(e_pulse,t_pulse, initial=0)
    # shift = 0.9
    # A_pump = scipy.interpolate.interp1d(np.concatenate([[-1e4],t_pulse-shift*2*np.pi,[1e4]]),A0*np.concatenate([[0],a_pulse,[a_pulse[-1]]]),kind='linear')
    # A_probe = scipy.interpolate.interp1d(np.concatenate([[-1e4],t_pulse-shift*2*np.pi+t_delay,[1e4]]),A0_pr*np.concatenate([[0],a_pulse,[a_pulse[-1]]]),kind='linear')

    def A(t):
        # return A_pump(t) + A_probe(t)
        """ Returns the vector potential at time t """
        return A0*np.exp(-(t-te)**2/(2*tau**2))*np.cos(w*t) \
            +  A0_pr*np.exp(-(t-te-t_delay)**2/(2*tau_pr**2))*np.cos(w_pr*(t-t_delay))


#%%
    def sprime(t, s):
        """ function given to the integrator used to calculate the time evolution of the state vector in the second order calculation """
        ds = np.copy(s).reshape(nb, ne_)
        # unpack and reshape all variables from s
        r11 = ds[:, :Ne]
        r21o = ds[:, Ne:2*Ne]
        r21e = ds[:, 2*Ne:3*Ne]
        R11 = ds[:, 3*Ne:(3*Ne+Ne**2)].reshape(nb,Ne,Ne)
        R21 = ds[:, (3*Ne+Ne**2):(3*Ne+2*Ne**2)].reshape(nb,Ne,Ne)
        f11 = ds[:, (3*Ne+2*Ne**2):(3*Ne+3*Ne**2)].reshape(nb,Ne,Ne)
        f21 = ds[:, (3*Ne+3*Ne**2):(3*Ne+4*Ne**2)].reshape(nb,Ne,Ne)

        # symmetrize
        # r11 = (r11 + np.flip(r11, axis=1))/2
        # r21e = (r21e + np.flip(r21e, axis=1))/2
        # r21o = (r21o - np.flip(r21o, axis=1))/2

        r21 = r21o+r21e

        # self-consitent recalculation of the gap
        d = U @ ( N0*integ(-d_eq1/E1*r11 + u1**2*r21 - v1**2*np.conj(r21), axis=1) )

        # this is for python broadcasting convenience
        d1 = d[:,ax]
        d2 = d[:,ax, ax]

        # first order
        d_f11 = -1j * (Ep - E) * f11 - 1j * (fp - f) * A(t)
        d_f21 = -1j * (Ep + E) * f21 + 1j * (1 - fp - f) * A(t)

        # second order
        d_r11 =  -2*A(t)*(e_charge*vf1)**2/3.0 * integ((l**2*f11.imag - p**2 * f21.imag) * W, axis=2)

        d_r21o = -1j*2*E1 * r21o + 1j*(1-2*f1)*(e1/E1) * d1.real \
            + 1j*2*A(t)*(e_charge*vf1)**2/3.0 * integ(W * l * p * ( f21 - np.conj(f11)), axis=2)

        d_r21e = -2*1j*E1 * r21e - 1j*(1-2*f1)/E1 * e_charge**2 * A(t)**2 / 2 * d_eq1 * s1/m1 - (1-2*f1)*d1.imag

        # third order
        d_R11 = -1j*(Ep-E)*R11 - 1j*np.transpose(l, axes=(0,2,1)) * (A(t) * (r11[:,ax,:] - r11[:,:,ax]) + d2.real * (d_eq/Ep - d_eq/E) * f11 ) \
            - 1j * np.transpose(p, axes=(0,2,1)) * (A(t) * (r21o[:,ax,:] + r21o[:,:,ax].conj()) + d2.real * (-ep/Ep * f21.conj() - e/E * f21) )

        d_R21 = -1j*(Ep+E)*R21 + 1j*np.transpose(p, axes=(0,2,1)) * (A(t) * (r11[:,ax,:] + r11[:,:,ax]) + d2.real * (d_eq/Ep + d_eq/E) * f21 ) \
            - 1j * np.transpose(l, axes=(0,2,1)) * (A(t) * (r21o[:,ax,:] - r21o[:,:,ax])       + d2.real * ( e/E * f11 - ep/Ep * f11.conj()) )


        # reshape and repack into ds
        ds[:, :Ne] = d_r11
        ds[:, Ne:2*Ne] = d_r21o
        ds[:, 2*Ne:3*Ne] = d_r21e
        ds[:, 3*Ne:(3*Ne+Ne**2)] = d_R11.reshape(nb,Ne**2)
        ds[:, (3*Ne+Ne**2):(3*Ne+2*Ne**2)] = d_R21.reshape(nb,Ne**2)
        ds[:, (3*Ne+2*Ne**2):(3*Ne+3*Ne**2)] = d_f11.reshape(nb,Ne**2)
        ds[:, (3*Ne+3*Ne**2):(3*Ne+4*Ne**2)] = d_f21.reshape(nb,Ne**2)

        return ds.reshape((ne,))

    start = time.time()

    # initial conditions - everything is zero since we start in equilibrium
    s0 = np.zeros(ne, dtype=complex)

    t0 = tmin
    tt = np.array_split(t_points,len(t_points)//10)

    JP1 = []
    JD1 = []
    D2 = []
    EFIELD = []
    JP3 = []
    JD3 = []

    for ts in tt:
        if task_ID==-1:
            print(np.round((ts[0]-tmin)/(tmax-tmin)*100))
        # the built in integrator solves for the r values numerically:
        sols = integrate.solve_ivp( sprime, (t0, ts[-1]), s0, t_eval=ts)
        t0 = ts[-1]
        s0 = np.copy(sols.y[:,-1])

        # extracting the solutions from the solver output:
        Y = sols.y.reshape(nb, ne_, len(sols.t)).swapaxes(0,2).swapaxes(1,2)
        t = sols.t
        r11 = Y[:, :, :Ne]
        r21o = Y[:, :, Ne:2*Ne]
        r21e = Y[:, :, 2*Ne:3*Ne]
        r21 = r21o + r21e
        R11 = Y[:, :, 3*Ne:(3*Ne+Ne**2)].reshape(len(t),nb,Ne,Ne)
        R21 = Y[:, :, (3*Ne+Ne**2):(3*Ne+2*Ne**2)].reshape(len(t),nb,Ne,Ne)
        F11 = Y[:, :, (3*Ne+2*Ne**2):(3*Ne+3*Ne**2)].reshape(len(t),nb,Ne,Ne)
        F21 = Y[:, :, (3*Ne+3*Ne**2):(3*Ne+4*Ne**2)].reshape(len(t),nb,Ne,Ne)


        # compute first order currents
        jp_1 = e_charge**2*np.sum((n/m) * integ(W*(l**2 * F11.real + p**2 * F21.real), axis=(2,3)), axis=1)

        eep = (e - ep) * np.ones((nb,1,1))
        for i in np.arange(nb):
            np.fill_diagonal(eep[i], 1)
        jd_1 = A(t) * e_charge**2 * np.sum((n/m) * integ( (ff-ffp)  / eep * W, axis=(1,2)))

        # compute e field
        """
        efield = A0*np.exp(-t**2/(2*tau**2))\
            *(t/tau**2*np.cos(w*t) + w*np.sin(w*t))\
                +A0_pr*np.exp(-(t-t_delay)**2/(2*tau**2)) \
            * ((t-t_delay)/tau**2*np.cos(w*(t-t_delay)) + w*np.sin(w*(t-t_delay)))
        """
        efield = -diff(t, A(t))


        # compute the second order gap from above solutions for all times
        d_2 = np.einsum('ij,tj->ti', U, N0*integ(-d_eq1/E1*r11 + u1**2*r21 - v1**2*np.conj(r21), axis=2))

        # compute third order current
        jp_3 = e_charge**2/2 * np.sum(n/m * integ(W * (l * (R11 + R11.conj()) + p * (R21 + R21.conj())), axis=(2,3)), axis=1)

        jd_3 = - e_charge**2 * A(t) * np.sum(s*N0/m * integ((u1**2 - v1**2) * (r11 + r11.conj() ) + 2*u1*v1 * (r21 + r21.conj()), axis=2) )

        JD1.append(jd_1)
        JP1.append(jp_1)
        D2.append(d_2)
        EFIELD.append(efield)
        JD3.append(jd_3)
        JP3.append(jp_3)

        del Y, r11, r21o, r21e, R11, R21, F11, F21, sols, jd_1, jp_1, d_2, jp_3, jd_3, efield
        gc.collect()

    jd_1 = np.concatenate(JD1)
    jp_1 = np.concatenate(JP1)
    d_2 = np.concatenate(D2)
    efield = np.concatenate(EFIELD)
    jd_3 = np.concatenate(JD3)
    jp_3 = np.concatenate(JP3)
    t = t_points
    tp = t/(2*np.pi)

    j_1 = jd_1 + jp_1
    j_3 = jp_3 + jd_3

    end = time.time()
    duration = end - start
    # pr.disable()
    print(f' finished in {duration}s')
    # pr.print_stats(sort='time')

#%% plot the data

    def pfft(t,data):
        N_t = len(t)
        y= data
        yf = fft(y)
        xf = fftfreq(N_t, t[1]-t[0])
        xf = fftshift(xf)
        yplot = fftshift(yf)
        return np.array([xf*2*np.pi, 1.0/N_t*yplot])

    def rfft(t,f, inverse=False):
        T = max(t) - min(t)
        Nt = len(t)
        dt = T/Nt
        xw = np.arange(Nt) * 2 * np.pi / T

        if inverse:
            xw = np.concatenate([xw[:Nt//2]+2*np.pi/dt, xw[Nt//2:]])
            idx = np.argsort(xw)
            xw = xw[idx]
            f = f[idx]
            fw = scipy.ifft(f, axis=0)/np.sqrt(Nt)*Nt
        else:
            fw = scipy.fft(f, axis=0)/np.sqrt(Nt)
            xw = np.concatenate([xw[:Nt//2], xw[Nt//2:]-2*np.pi/dt])
            idx = np.argsort(xw)
            xw = xw[idx]
            fw = fw[idx]
        return xw, fw

    def nm(x):
        return x / np.max(np.abs(x))

    if job_ID == -1:

        plt.figure('A')
        plt.clf()
        plt.subplot(131)
        plt.plot(tp,A(t))
        plt.plot(tp,efield)
        # plt.xlim((-1,1))
        plt.ylabel(f'$A(t)$')
        plt.xlabel(f'$t/2 \pi$')

        plt.subplot(132)
        tw, aw = rfft(t,A(t))
        # plt.plot(tw, np.real(nm(aw)))
        # plt.plot(tw, np.imag(nm(aw)))
        plt.plot(tw, np.abs(nm(aw)),'-')
        plt.xlim((0,5*d_eq0[0]))
        plt.ylabel(f'$A(\omega)$')
        plt.xlabel(f'$\omega$')
        plt.axvline(d_eq[0], c='gray', lw=1)
        plt.xlim((0,5*d_eq[0]))
        if len(d_eq)>1: plt.axvline(d_eq[1], c='gray', lw=1)
        plt.tight_layout()

        plt.subplot(133)
        tw, aw2 = rfft(t,A(t)**2)
        # plt.plot(tw, np.real(nm(aw2)))
        # plt.plot(tw, np.imag(nm(aw2)))
        plt.plot(tw, np.abs(nm(aw2)),'-')
        plt.xlim((0,5*d_eq0[0]))
        plt.ylabel(f'$A^2(\omega)$')
        plt.xlabel(f'$\omega$')
        plt.axvline(2*d_eq[0], c='gray', lw=1)
        plt.xlim((0,5*d_eq[0]))
        if len(d_eq)>1: plt.axvline(2*d_eq[1], c='gray', lw=1)
        plt.tight_layout()

        plt.figure('d_1')
        plt.clf()
        plt.subplot(121)
        plt.plot(tp,d_2.real)
        plt.title(f'Re$[\Delta]$')
        plt.subplot(122)
        sel = np.abs(tp-7)<4
        dsel = d_2[sel,0].real
        dsel -= np.mean(dsel)
        tw, dw = rfft(t[sel],dsel)
        plt.plot(tw, nm(np.abs(dw)), '.-')
        tw, dw = rfft(t[sel],dsel)
        plt.plot(tw, nm(np.abs(dw)), '.-')
        plt.xlim((0,5*d_eq0[0]))
        # plt.plot(tp, d_2.imag)
        # plt.title(f'Im$[\Delta]$')
        plt.tight_layout()


        plt.figure('sigma_1')
        plt.clf()

        #calculate conductivity
        def cond(j_1):
            tw, jw = rfft(t,j_1)
            tw, ew = rfft(t,efield)
            s=jw/ew
            ww, Aw = rfft(t,A(t))
            s2 = jw/(1j*ww*Aw)
            s[tw==0] = 0 + 1j*np.inf
            s2[tw==0] = 0 + 1j*np.inf
            sr = np.abs(s.real)
            si = np.abs(s.imag)
            sr2 = np.abs(s2.real)
            si2 = np.abs(s2.imag)
            wsel = np.abs(tw-2)<0.3
            nm1 = np.max(sr[wsel])
            nm2 = np.max(si[wsel])
            plt.plot(tw, sr/nm1, label='Re $j/E$')
            plt.plot(tw, si/nm2, label='Im $j/E$')
            plt.plot(tw, sr2/nm1, '.', c='blue', label='Re $j/i\omega A$')
            plt.plot(tw, si2/nm2, '.', c='orange', label='Im $j/i \omega A$')
            plt.xlim((0,4*d_eq0[0]))
            plt.ylim(0,10)
            plt.legend()

        plt.subplot(131)
        cond(j_1)
        plt.subplot(132)
        cond(j_3)
        plt.subplot(133)
        cond(j_1+0.1*j_3)



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
            'jp_1': jp_1,
            'jd_1': jd_1,
            'd_2': d_2,
            'jp_3': jp_3,
            'jd_3': jd_3,
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

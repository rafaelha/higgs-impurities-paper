#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
import scipy
from scipy.fftpack import fft, fftfreq, fftshift

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
        fw = scipy.ifft(f, axis=0)/np.sqrt(Nt)*Nt
    else:
        fw = scipy.fft(fftshift(f), axis=0)/np.sqrt(Nt)
        xw = np.concatenate([xw[:Nt//2], xw[Nt//2:]-2*np.pi/dt])
        idx = np.argsort(xw)
        xw = xw[idx]
        fw = fw[idx]
    return xw, fw

# 1 time unit = 6.58285E-2 ps
u_t = 6.58285E-2
u_e = 10
meV_to_THz = 0.2417990504024

d0 = 0.13

ps = np.loadtxt('pulse_shape.csv', delimiter=',')
t_pulse = ps[:,0]
e_pulse = ps[:,1]
e_pulse /= np.max(np.abs(e_pulse))
t_pulse /= u_t

x = t_pulse
y = e_pulse
plt.figure('data')
plt.clf()
plt.plot(x,y)


def Efield(t, tau, te, dt, w, A0):
    # return A_pump(t) + A_probe(t)
    """ Returns the vector potential at time t """
    return A0*np.exp(-(t-te-dt)**2/(2*tau**2)) \
        * ( (t-te-dt)/tau**2 * np.cos(w*(t-dt)) + w * np.sin(w*(t-dt)) )

def A(t):
# return A_pump(t) + A_probe(t)
    """ Returns the vector potential at time t """
    return A0*np.exp(-(t-te)**2/(2*tau**2))*np.cos(w*t) # \
        # +  A0_pr*np.exp(-(t-te-t_delay)**2/(2*tau_pr**2))*np.cos(w_pr*(t-t_delay))


tau = 5.37
te = -40
dt = 100
dt = 0
w = 0.04
A0 = -25

t = np.linspace(-75, max(t_pulse), 1000)
plt.plot(t, Efield(t, tau, te, dt, w, A0))
E_fit = Efield(t, tau, te, dt, w, A0)
plt.xlim((min(t), max(t)))

#%% fig 1 - pulse
plt.rcParams.update({'font.size': 13})
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

plt.figure('pulse-t', figsize=(4.5,1.8))
plt.clf()
t = np.linspace(-75, 500, 1000)
plt.plot(t*u_t, Efield(t, tau, te, dt, w, A0), 'k')
plt.xlim((-4.5,20))
plt.ylabel('$E(t)$')
plt.xlabel('$t$ (ps)')
plt.tight_layout()
plt.savefig('pulse.pdf', transparent=True)

plt.figure('pulse-w', figsize=(2.1,1.8))
plt.clf()
ww, ew = fft(t, A(t)**2)
plt.plot(ww/u_t/(2*np.pi),np.abs(ew), 'k')
plt.xlim((0,3))
# plt.axvline(d0*u_e*meV_to_THz, c='k', lw=0.7)
plt.axvline(2*d0*u_e*meV_to_THz, c='k', lw=0.3)
plt.xlabel('Frequency (THz)')
plt.ylabel('$|A^2(\omega)|$')
plt.tight_layout()
plt.savefig('pulse_w.pdf', transparent=True)

plt.figure('pulse1-w', figsize=(2.1,1.8))
plt.clf()
ww, ew = fft(t, A(t))
plt.plot(ww/u_t/(2*np.pi),np.abs(ew), 'k')
plt.xlim((0,3))
# plt.axvline(d0*u_e*meV_to_THz, c='k', lw=0.7)
plt.axvspan(2*d0*u_e*meV_to_THz, 5, facecolor='gray', alpha=0.3)
plt.xlabel('Frequency (THz)')
plt.ylabel('$|A(\omega)|$')
plt.tight_layout()
plt.savefig('pulse1_w.pdf', transparent=True)
#%%
from scipy.optimize import curve_fit

popt, pcov = curve_fit(Efield, x,y, p0=[tau, te, dt, w, A0])
plt.plot(t, Efield(t,*popt))

plt.figure()
plt.plot(t*u_t, Efield(t,*popt))
plt.xlabel('t (ps)')

plt.figure()

plt.subplot(121)
ww, ew = fft(t, E_fit)
plt.plot(ww/u_t/(2*np.pi),np.abs(ew)**2)
plt.xlim((0,2))
plt.axvline(d0*u_e*meV_to_THz, c='k', lw=0.7)
plt.axvline(2*d0*u_e*meV_to_THz, c='k', lw=0.3)
plt.xlabel('Frequency (THz)')
plt.ylabel('Intensity $(E(\omega))^2$')

plt.subplot(122)
ww, ew = fft(t, E_fit**2)
plt.plot(ww/u_t/(2*np.pi),np.abs(ew))
plt.xlim((0,2))
plt.axvline(d0*u_e*meV_to_THz, c='k', lw=0.3)
plt.axvline(2*d0*u_e*meV_to_THz, c='k', lw=0.7)
plt.xlabel('Frequency (THz)')
plt.ylabel('Intensity $E^2(\omega)$')
plt.tight_layout()
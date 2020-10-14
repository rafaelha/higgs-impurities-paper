#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
import scipy
from scipy.fftpack import fft, fftfreq, fftshift

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
        fw = scipy.fft(fftshift(f), axis=0)/np.sqrt(Nt)
        xw = np.concatenate([xw[:Nt//2], xw[Nt//2:]-2*np.pi/dt])
        idx = np.argsort(xw)
        xw = xw[idx]
        fw = fw[idx]
    return xw, fw

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def nm(x):
    return x / np.max(np.abs(x))

ps = np.loadtxt('..\\material_parameters_Matsunaga\\pulse_shape.csv', delimiter=',')
rm = 20

t_pulse = running_mean(ps[:,0],rm)

b = 0.3
smooth = (np.tanh((t_pulse-1.5)/b)+1)/2
smooth2 = (-np.tanh((t_pulse-7.5)/b)+1)/2

e_pulse = running_mean(ps[:,1],rm)*smooth*smooth2
e_pulse /= np.max(np.abs(e_pulse))

# 1 time unit = 0.438856 ps
t_pulse /= 0.438856

A0=1
a_pulse = -scipy.integrate.cumtrapz(e_pulse,t_pulse, initial=0)
shift = 0
A_pump = scipy.interpolate.interp1d(np.concatenate([[-1e6],t_pulse-shift*2*np.pi,[1e6]]),A0*np.concatenate([[0],a_pulse,[a_pulse[-1]]]),kind='linear')
A_probe = scipy.interpolate.interp1d(np.concatenate([[-1e6],t_pulse-shift*2*np.pi+t_delay,[1e6]]),A0_pr*np.concatenate([[0],a_pulse,[a_pulse[-1]]]),kind='linear')

A0 = -1.2
te=-1.3
tau = 0.28*np.pi
w = 0.85
A0_pr = 0
tau_pr = 0
w_pr = 0
t_delay = 0
ii = 0
t = np.linspace(-100*np.pi+ii,100*np.pi-ii,1000)
tp = t/2/np.pi
dt = 10.35


tau = 0.81
te = -4.21
dt = 13.3
w = 0.36
A0 = -2.73


def A(t):
    # return A_pump(t) + A_probe(t)
    """ Returns the vector potential at time t """
    return A0*np.exp(-(t-te-dt)**2/(2*tau**2))*np.cos(w*(t-dt)) \
        +  A0_pr*np.exp(-(t-t_delay)**2/(2*tau_pr**2))*np.cos(w_pr*(t-t_delay))
def A_exp(t):
    return A_pump(t) + A_probe(t)


plt.figure('A')
plt.clf()
plt.subplot(131)
plt.plot(tp,A(t))
plt.plot(tp,A_exp(t),'--')
# plt.xlim((-1,1))
plt.ylabel(f'$A(t)$')
plt.xlabel(f'$t/2 \pi$')

plt.subplot(132)
tw, aw = rfft(t,A(t))
plt.plot(tw, np.real(nm(aw)))
plt.plot(tw, np.imag(nm(aw)))
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
plt.plot(tw, np.real(nm(aw2)))
plt.plot(tw, np.imag(nm(aw2)))
plt.plot(tw, np.abs(nm(aw2)),'-')
plt.xlim((0,5*d_eq0[0]))
plt.ylabel(f'$A(\omega)$')
plt.xlabel(f'$\omega$')
plt.axvline(2*d_eq[0], c='gray', lw=1)
plt.xlim((0,5*d_eq[0]))
if len(d_eq)>1: plt.axvline(2*d_eq[1], c='gray', lw=1)
plt.tight_layout()

plt.figure('E')
plt.clf()
plt.subplot(121)
efield = -np.diff(A(t))/(t[1]-t[0])
efield_exp = -np.diff(A_exp(t))/(t[1]-t[0])
efield = np.concatenate([[efield[0]],efield])
efield_exp = np.concatenate([[efield_exp[0]],efield_exp])
plt.plot(tp,efield)
plt.plot(tp,efield_exp)
plt.xlim((-1,1))
plt.ylabel(f'$E(t)$')

plt.xlabel(f'$t/2 \pi$')

plt.subplot(122)
tw, ew = rfft(t,efield)
tw, ew2 = rfft(t,A(t))
ew2 = -1j*tw*ew2
plt.plot(tw, np.real(nm(ew)))
plt.plot(tw, np.imag(nm(ew)))
plt.plot(tw, np.abs(nm(ew)),'-')
plt.plot(tw, np.real(nm(ew2)),'--')
plt.plot(tw, np.imag(nm(ew2)),'--')
plt.plot(tw, np.abs(nm(ew2)),'--')
plt.xlim((0,5*d_eq0[0]))
plt.ylabel(f'$E(\omega)$')
plt.xlabel(f'$\omega$')
plt.axvline(d_eq[0], c='gray', lw=1)
plt.xlim((0,5*d_eq[0]))
if len(d_eq)>1: plt.axvline(d_eq[1], c='gray', lw=1)
plt.tight_layout()

#%%
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


tau = 0.81
te = -4.21
dt = 13.3
w = 0.36
A0 = -2.73

plt.plot(t_pulse, Efield(t_pulse, tau, te, dt, w, A0))

from scipy.optimize import curve_fit

popt, pcov = curve_fit(Efield, x,y, p0=[tau, te, dt, w, A0])
plt.plot(x, Efield(x,*popt))
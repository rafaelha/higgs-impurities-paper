#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import pickle


hbar=1
kb=1
e=1

u_temp = 116.032
u_en = 10
meV_to_THz = 0.2417990504024

pre_d0  = np.array([0.3,0.7])



wd  = 5
s = np.array([1,-1])
m = np.array([0.85, 1.38])
ef = np.array([290, 70])
v_legget = 0.15


T=4/u_temp
B = 1/(kb*T)

nb = len(m)
kf = np.sqrt(2*m*ef)/hbar
vf = kf/m
n = kf**3/(3*np.pi**2)
N0 = m*kf/(2*np.pi**2)

Ne = 1000
ep = np.linspace(-wd, wd, Ne)

#%%
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

def find_U(v, d0, I):
    # this probably isn't valid as it results in U12 neq U21, but i was trying to investigate the v= infty (intraband=0) case...
    if v == np.inf:
        return np.array([[0, (d0[1]/d0[0]*N0[1]*integrate.simps(np.tanh(B/2*np.sqrt(ep**2+d0[1]**2))/(2*np.sqrt(ep**2+d0[1]**2)), ep))**-1],
                        [(d0[0]/d0[1]*N0[0]*integrate.simps(np.tanh(B/2*np.sqrt(ep**2+d0[0]**2))/(2*np.sqrt(ep**2+d0[0]**2)), ep))**-1, 0]])

    # regular calculation of remaining U parameters:
    if nb == 2:
        U = np.array([[1, 1],
                    [1, 1]])

        U11 = d0[0]/(N0[0]*d0[0]*I[0]+v*N0[1]*d0[1]*I[1])
        U22 = (d0[1]-v*U[0, 0]*N0[0]*d0[0]*I[0])/(N0[1]*I[1]*d0[1])
        U12 = v*U11
        U_new = np.array([[U11, U12],
                        [U12, U22]])

        while (U_new != U).all():
            U = U_new
            U11 = d0[0]/(N0[0]*d0[0]*I[0]+v*N0[1]*d0[1]*I[1])
            U22 = (d0[1]-v*U[0, 0]*N0[0]*d0[0]*I[0])/(N0[1]*I[1]*d0[1])
            U12 = v*U11
            U_new = np.array([[U11, U12],
                            [U12, U22]])
        return U_new
    else:
        return("Leggett mode only for multiband")

B = 1/(kb*0.000001)
ep = np.linspace(-wd, wd, Ne)
U = find_U(v_legget, pre_d0, integrate.simps(0.5*1/np.sqrt(ep**2+pre_d0.reshape(2,1)**2)*np.tanh(B/2*np.sqrt(ep**2+pre_d0.reshape(2,1)**2)),ep))
UN0 = U*N0[:, np.newaxis]
print('U=',U)
print('UN0=',UN0)
d_eq0_T0 = find_d0(UN0)
print('gap=',d_eq0_T0, 'at T=0')









#%%


temps = np.linspace(0,50/u_temp,250)
gaps = []

for T in temps:
    B = 1/(kb*T)
    d_eq0 = find_d0(UN0)
    print('gap=',d_eq0, 'at T=',T*u_temp)
    gaps.append(d_eq0)

#%%
gaps = np.stack(gaps)
plt.figure(figsize=(3,2))
plt.plot(temps*u_temp, 2*gaps*u_en*meV_to_THz, 'k')
plt.ylim((0,np.max(pre_d0)*2*1.2*u_en*meV_to_THz))
plt.xlabel('T (K)')
plt.ylabel('$2\Delta$ (THz)')
plt.tight_layout()
plt.savefig('BCS.pdf')
import pickle
f1 = open(f'BCS.pickle', 'ab')
pickle.dump(temps*u_temp, f1)
pickle.dump(2*gaps*u_en*meV_to_THz, f1)
f1.close()
# plt.xlim((8,9))

#%%
f1 = open(f'BCS.pickle', 'rb')
a=pickle.load(f1)
b=pickle.load(f1)
f1.close()

plt.figure(figsize=(1.8,1.3))
plt.plot(a, b, 'k')
plt.ylim((0,4))
plt.xlim((0,50))
plt.xlabel('T (K)')
plt.ylabel('$2\Delta$ (THz)')
plt.tight_layout()
plt.savefig('BCS-1.pdf', transparent=True)
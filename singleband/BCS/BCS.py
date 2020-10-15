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

d0 = np.array([0.13])
# U = np.array([[0.46]])
# d0  = np.array([0.20])
# U = np.array([[0.5]])
# d0  = np.array([0.23])
# U = np.array([[0.525]])

us = [np.array([[0.459]]), np.array([[0.5]]), np.array([[0.524]])]

m=np.array([0.78])
ef=np.array([100])
wd = 5.5


s=np.array([1])
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

def d0_integrand(x, d):
    # This is an auxiliary function used in find_d0 to calculate an integral
    return 0.5*1/np.sqrt(x**2+d**2)*np.tanh(B/2*np.sqrt(x**2+d**2))


def find_d0(UN0, d_last):
    alpha = 0.05
    # this function finds the initial gap(s) given U*N(0). Works in the single or multiband case
    if nb == 2:
        d = np.array([1, 1])
        integral = np.zeros(2)
        for j in [0, 1]:
            integral[j] = integrate.quad(d0_integrand, -wd, wd, (d[j],))[0]
        d_new = np.sum(UN0*d*integral, axis=1)

        while (d != d_new).all(): # seems like convergence of gap is based on machine precision
            d = d_new
            integral = np.zeros(2)
            for j in [0, 1]:
                integral[j] = integrate.quad(d0_integrand, -wd, wd, (d[j],))[0]
            d_new = np.sum(UN0*d*integral, axis=1)
        return d_new
    elif nb == 1:
        d = d_last
        d_new = UN0*d*integrate.quad(d0_integrand, -wd, wd, (d,))[0]
        while np.abs(d-d_new) > 1e-7:
            d = alpha * d_new + (1-alpha) * d
            d_new = UN0*d*integrate.quad(d0_integrand, -wd, wd, (d,))[0]
        return d_new
uu = 1 / N0 / integrate.quad(d0_integrand, -wd, wd, (d0[0],))[0]
print(uu)

UN0 = np.array([[uu]])*N0[:, np.newaxis]

d02 = find_d0(UN0, 1)
print(d0,d02)

temps = np.linspace(0,17/u_temp,100)
gaps = np.zeros((temps.shape[0],3))
for j in np.arange(3):
    UN0 = us[j]*N0[:, np.newaxis]
    i = 0
    d_last = d0[0]
    for T in temps:
        B = 1/(kb*T)
        d_last = find_d0(UN0, d_last)[0,0]
        gaps[i,j] = d_last
        print(j,i, T*u_temp, gaps[i])
        i += 1

#%%
plt.figure(figsize=(3,2))
plt.plot(temps*u_temp, 2*gaps*u_en*meV_to_THz, 'k')
plt.ylim((0,d0[0]*2*2*u_en*meV_to_THz))
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
plt.plot(a, b[:,0], 'k')
plt.ylim((0,0.8))
plt.xlim((0,10))
plt.xlabel('T (K)')
plt.ylabel('$2\Delta$ (THz)')
plt.tight_layout()
plt.savefig('BCS-1.pdf', transparent=True)
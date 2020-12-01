import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from matplotlib.collections import LineCollection


k = np.linspace(-1,1,1000)
e = 2*k
uk2 = 0.5*(1-e/E)
vk2 = 0.5*(1+e/E)
m = 0.4
E = np.sqrt(e**2+m**2)
# plt.tight_layout()
# plt.plot(k,E, 'r')
# plt.plot(k,-E, 'r')
# plt.plot(k,uk2)
# plt.plot(k,vk2)


colors = [(12/255, 35/255, 68/255), (142/255, 142/255, 142/255), (255/255, 0/255, 0/255)]  # R -> G -> B
cmap_name = 'my_list'
n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=300)

plt.clf()
fig, ax = plt.subplots(1,1, num="one")
fig.set_size_inches(1.5,2.5)
ax.set_frame_on(False)
plt.tick_params(left=False,bottom=False,labelbottom=False, labelleft=False) #remove ticks
plt.box(False) #remove box

ax.plot(k,e, 'r', alpha=0.08, lw=5.5)
ax.plot(k,-e, 'b', alpha=0.08, lw=5.5)

def plotLine(k,E,uk2):
    norm = plt.Normalize(0,1)
    points = np.array([k, E]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cdata = uk2
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(cdata)
    lc.set_linewidth(5.5)
    line = ax.add_collection(lc)

plotLine(k,E,vk2)
plotLine(k,-E,uk2)
plt.savefig('dispersion.pdf')
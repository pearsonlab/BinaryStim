import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.rcParams['text.usetex'] = True

sns.set_context('talk')
palette = sns.color_palette("GnBu_d")

plt.figure(figsize=(6,4))

i=0
for r in [0.01,0.05,0.1]:
    sens = []
    spec = []
    for thresh in [-.01,0.1,0.3,0.5,0.6,0.7,0.9,1,1.01]:
        sens.append(np.loadtxt(open('./finalfinal/sens_fit_N1k_T500_3_ab'+str(r)+'_thr'+str(thresh)+'.txt', 'rb'), delimiter=' '))
        spec.append(np.loadtxt(open('./finalfinal/spec_fit_N1k_T500_3_ab'+str(r)+'_thr'+str(thresh)+'.txt', 'rb'), delimiter=' '))

    sens = np.array(sens)
    spec = np.array(spec)
    
    sns.lineplot(1-spec[:,-1], sens[:,-1], label=r'$\alpha,\beta={}$'.format(r), color=palette[i])
    i+=1

r=0.2
sens = []
spec = []
for thresh in [-.01,0.1,0.3,0.5,0.6,0.7,0.9,1,1.01]:
    sens.append(np.loadtxt(open('./finalfinal/sens_fit_N1k_T500_3_50it_ab'+str(r)+'_thr'+str(thresh)+'.txt', 'rb'), delimiter=' '))
    spec.append(np.loadtxt(open('./finalfinal/spec_fit_N1k_T500_3_50it_ab'+str(r)+'_thr'+str(thresh)+'.txt', 'rb'), delimiter=' '))
sens = np.array(sens)
spec = np.array(spec)

g = sns.lineplot(1-spec[:,-1], sens[:,-1],  label=r'$\alpha,\beta={}$'.format(r), color=palette[i])


ax1 = plt.gca()
ax1.set_xlim(-0.02,1.02)
ax1.set_ylim(-0.02,1.02)

ax1 = plt.gca()
ax1.set_xlabel(r'False positive rate')
ax1.set_ylabel(r'True positive rate')
g.legend(loc='lower right')

# plt.tight_layout()

fig = plt.gcf()
fig.savefig('Fig2b.png', bbox_inches='tight', dpi=300)

plt.show(block=True)

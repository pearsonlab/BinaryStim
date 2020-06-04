import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.rcParams['text.usetex'] = True

sns.set_context('talk')
palette = sns.color_palette()
plt.figure(figsize=(6, 4))

tests = np.array([10,20,50,100,200,500,600,800,1000])
## batch
spec=[]
for T in tests:
    spec.append(np.loadtxt(open('./finalfinal/spec_fit_ab05_N1k_T'+str(T)+'.txt', 'rb'), delimiter=' '))
spec = np.array(spec)

sns.lineplot(tests,spec[:,-1], label=r'Batch Bernoulli',  color=palette[0])

## stream
spec=[]
tests = np.arange(10,1000)
spec.append(np.loadtxt(open('./finalstream/spec_stream_ab05_N1k_T1k.txt', 'rb'), delimiter=' '))
spec = np.squeeze(np.array(spec).T[:-1])
g = sns.lineplot(tests,spec, label=r'Online Bernoulli',  color=palette[1])


## adapt
spec=[]
spec.append(np.loadtxt(open('./finalstream/spec_adapt_ab05_N1k_T1k.txt', 'rb'), delimiter=' '))
spec = np.squeeze(np.array(spec).T[:-1])
sns.lineplot(tests,spec, label=r'Online adaptive',  color=palette[3])

ax1 = plt.gca()
ax1.set_xlabel('Tests')#, fontsize=12)
ax1.set_ylabel('Specificty')#, fontsize=12)
ax1.set_ylim(-0.02,1.02)
g.legend(loc='lower right')
# g1.legend().set_visible(False)
plt.tight_layout()

fig = plt.gcf()
fig.savefig('Fig3a.png', bbox_inches='tight', dpi=300)

plt.show()
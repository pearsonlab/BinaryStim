import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.rcParams['text.usetex'] = True

sns.set_context('talk')
palette = sns.color_palette("Reds",6) 
plt.figure(figsize=(6, 4))

i=1
for N in [100,200,500,1000,2000]: ## T is set to N
    T = np.arange(1,N)
    # sens = np.loadtxt(open('./finalstream/sens_stream_N'+str(N)+'.txt', 'rb'), delimiter=' ')[:-1]
    spec = np.loadtxt(open('./finalstream/spec_stream_N'+str(N)+'.txt', 'rb'), delimiter=' ')[:-1]
    sns.lineplot(T/T[-1], spec, label=r'$N={}$'.format(N), color=palette[i])
    i+=1

T = np.arange(1,3000)
spec = np.loadtxt(open('./finalstream/spec_adapt_ab05_N10k_T3k.txt', 'rb'), delimiter=' ')[:-1]
g = sns.lineplot(T/10000, spec, label=r'$N=10000$', color='black')
    
ax1 = plt.gca()
ax1.set_xlabel('Tests / Number of neurons')#, fontsize=12)
ax1.set_ylabel('Specificity')#, fontsize=12)
ax1.set_ylim(-0.02,1.02)
ax1.set_xlim(-0.02,1.02)
# ax1.legend()
g.legend(loc='lower right')
# g1.legend().set_visible(False)
plt.tight_layout()


fig = plt.gcf()
fig.savefig('Fig3b.png', bbox_inches='tight', dpi=300)

plt.show()
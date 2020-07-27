import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.rcParams['text.usetex'] = True

sns.set_context('talk')
palette = sns.color_palette()
plt.figure(figsize=(6,4))


w = []

for T in [100,500,1000, 1500, 2000]: 
    w.append(np.loadtxt(open('./final/save_w_ab01_'+str(T)+'.txt', 'rb'), delimiter=' '))
w = np.array(w).T

w0 = np.loadtxt(open('./final/save_w_gt_N1k_1000.txt', 'rb'), delimiter=' ')

gt_TP = w0>0  #locations of all true positives
w_P = w>0.5
w_FP = np.logical_and(np.any(w_P,axis=1), ~gt_TP)
w_plot = np.logical_or(gt_TP,w_FP)

d = {'100':w[w_plot,0],'500':w[w_plot,1],'1000':w[w_plot,2],'1500':w[w_plot,3],'2000':w[w_plot,4],'ID':gt_TP[w_plot]}

df = pd.DataFrame(d)
df.ID = df.ID.replace({True:'TP', False:'FP'})
df = pd.melt(df, 'ID', var_name='Weights')
g = sns.stripplot(y='value', x='Weights', hue='ID', data=df, palette=[palette[0],palette[1]], jitter=1, \
    dodge=True, orient='v', order=(['100','500','1000','1500','2000']), size=3, alpha=0.5)
g.axhspan(-0.03, 0.03, color='black', alpha=0.1)
g.axhspan(0.97, 1.03, color=palette[0], alpha=0.2)
g.axhline(y=0.5, lw=2, clip_on=False, color='black', linestyle='--')
g.set_ylabel(r'Weights')
g.set_xlabel(r'Tests')

g.legend(loc='upper left', bbox_to_anchor=(0,1.02), handletextpad=0.05, borderpad=0.2, labelspacing=0.25) #,0.5,0.4))  bbox_to_anchor=(-0.1,1), 

# plt.tight_layout()

fig = plt.gcf()
fig.savefig('Fig2a.png', bbox_inches='tight', dpi=300)

plt.show()
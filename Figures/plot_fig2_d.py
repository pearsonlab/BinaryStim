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

spec = []
spec5 = []
spec10 = []
tests = np.array([10,20,50,100,200,500,600,800,1000,1200,1500,2000,2500])

for T in tests:
    spec.append(np.loadtxt(open('./finalfinal/spec_fit_ab01_N1k_T'+str(T)+'.txt', 'rb'), delimiter=' '))
    spec5.append(np.loadtxt(open('./finalfinal/spec_fit_ab05_N1k_T'+str(T)+'.txt', 'rb'), delimiter=' '))
    spec10.append(np.loadtxt(open('./finalfinal/spec_fit_ab10_N1k_T'+str(T)+'.txt', 'rb'), delimiter=' '))
spec = np.array(spec)
spec5 = np.array(spec5)
spec10 = np.array(spec10)

sns.lineplot(tests,spec[:,-1], label=r'Bernoulli, $\alpha,\beta=0.01$',  color=palette[0])
sns.lineplot(tests,spec5[:,-1], label=r'Bernoulli, $\alpha,\beta=0.05$',  color=palette[0], linestyle='--')
sns.lineplot(tests,spec10[:,-1], label=r'Bernoulli, $\alpha,\beta=0.10$',  color=palette[0], linestyle='-.')

### naive
T=2500
naive_spec = np.loadtxt(open('./finalfinal/spec_naive_ab01_'+str(T)+'.txt', 'rb'), delimiter=' ')
naive_tests = np.arange(0,2500)
sns.lineplot(naive_tests,naive_spec,  label=r'Naive, $\alpha,\beta=0.01$', color=palette[2])
naive_spec = np.loadtxt(open('./finalfinal/spec_naive_ab05_'+str(T)+'.txt', 'rb'), delimiter=' ')
sns.lineplot(naive_tests,naive_spec, label=r'Naive, $\alpha,\beta=0.05$', color=palette[2], linestyle='--')
naive_spec = np.loadtxt(open('./finalfinal/spec_naive_ab10_'+str(T)+'.txt', 'rb'), delimiter=' ')
g = sns.lineplot(naive_tests,naive_spec, label=r'Naive, $\alpha,\beta=0.10$', color=palette[2], linestyle='--')
g.lines[1].set_linestyle('--')
g.lines[4].set_linestyle('--')
g.lines[2].set_linestyle('-.')
g.lines[5].set_linestyle('-.')

ax1 = plt.gca()
ax1.set_xlabel(r'Tests')
ax1.set_ylabel(r'Specificity')
ax1.set_ylim(0,1.02)
ax1.set_xlim(-0.02,2500.02)
# g.legend().set_visible(False)
g.legend(loc='lower right')

fig = plt.gcf()
fig.savefig('Fig2d.png', bbox_inches='tight', dpi=300)

plt.show()
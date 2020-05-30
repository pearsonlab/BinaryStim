"""
Generate a figure illustrating bounds on entropy gradients and their magnitudes.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['text.usetex'] = True

# set up figure
sns.set_context('talk')
palette = sns.color_palette()
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# set up parameters

# range of data
x = np.linspace(0, 1, 500)[1:-2]

# max entropy points under independent model
w0 = 1/2

# the max entropy of a depends on w, so let's choose some values
w = [0.5, 0.85, 0.70, 0.85, 0.8]
a0 = 1 - np.prod(w)

# strong convexity
sig = 4


############### Bounds for w ###############

# strong convexity lower bound
grad_H_sc = -sig * (x - w0)

# independence lower bound
grad_H_ind = -np.log(x) + np.log(1-x)

# constraint-based bounds
a, b = 0.2, 0.6  # just illustrative values
grad_H_cons_lb = grad_H_ind + np.log(a) - np.log(1-a)
grad_H_cons_ub = grad_H_ind + np.log(b) - np.log(1-b)

# truncate last two based on constraints so they're not plotted outside
# valid regions
x_cons_valid = np.logical_and(x >= a, x <= b)

############### Plot 1: w entropy gradients ###############
ax[0, 0].plot(x, grad_H_sc, label='Strong Convexity')
ax[0, 0].plot(x, grad_H_ind, label = r'Independent $w$')
ax[0, 0].plot(x[x_cons_valid], grad_H_cons_lb[x_cons_valid], color=palette[2],
         linestyle='--')
ax[0, 0].plot(x[x_cons_valid], grad_H_cons_ub[x_cons_valid], color=palette[2],
         label='Primal Feasible')
ax[0, 0].axvspan(a, b, color='black', alpha=0.1)

ax[0, 0].set_xlabel(r"$w$")
ax[0, 0].set_ylabel(r"$\nabla\mathcal{H}$")
ax[0, 0].legend()
ax[0, 0].set_xticks([0, 0.5, 1])


############### Plot 3: absolute w entropy gradients ###############
ax[1, 0].plot(x, np.abs(grad_H_sc), label='Strong Convexity')
ax[1, 0].plot(x, np.abs(grad_H_ind), label=r'Independent $w$')
abs_grad_H_cons_ub = np.maximum(np.abs(grad_H_cons_lb), np.abs(grad_H_cons_ub))
abs_grad_H_cons_lb = np.minimum(np.abs(grad_H_cons_lb), np.abs(grad_H_cons_ub))

ax[1, 0].plot(x[x_cons_valid], abs_grad_H_cons_lb[x_cons_valid],
         color=palette[2], linestyle='--')
ax[1, 0].plot(x[x_cons_valid], abs_grad_H_cons_ub[x_cons_valid],
         color=palette[2], label='Primal Feasible')
ax[1, 0].axvspan(a, b, color='black', alpha=0.1)

ax[1, 0].set_xlabel(r"$w$")
ax[1, 0].set_ylabel(r"$|\nabla\mathcal{H}|$")
ax[1, 0].set_xticks([0, 0.5, 1])


############### Bounds for a ###############
grad_H_sc = -sig * (x - a0)

# independence lower bound
grad_H_ind = -np.log(x/a0) + np.log((1-x)/(1-a0))

# constraint-based bounds
a, b = np.max(w), np.minimum(np.sum(w), 1 - 1e-6)

grad_H_cons_lb = grad_H_ind + np.log(a) - np.log(1-a)
grad_H_cons_ub = grad_H_ind + np.log(b) - np.log(1-b)

# truncate last two based on constraints
x_cons_valid = np.logical_and(x >= a, x <= b)


############### Plot 2: a entropy gradients ###############
ax[0, 1].plot(x, grad_H_sc, label='Strong Convexity')
ax[0, 1].plot(x, grad_H_ind, label = r'Independent $w$')
ax[0, 1].plot(x[x_cons_valid], grad_H_cons_lb[x_cons_valid], color=palette[2],
         linestyle='--')
ax[0, 1].plot(x[x_cons_valid], grad_H_cons_ub[x_cons_valid], color=palette[2],
         label='Primal Feasible')
ax[0, 1].axvspan(a, b, color='black', alpha=0.1)

ax[0, 1].set_xlabel(r"$a$")
ax[0, 1].set_ylabel(r"$\nabla\mathcal{H}$")
ax[0, 1].set_xticks([0, 0.5, 1])


############### Plot 4: a entropy gradients ###############

ax[1, 1].plot(x, np.abs(grad_H_sc), label='Strong Convexity')
ax[1, 1].plot(x, np.abs(grad_H_ind), label=r'Independent $w$')
abs_grad_H_cons_ub = np.maximum(np.abs(grad_H_cons_lb), np.abs(grad_H_cons_ub))
abs_grad_H_cons_lb = np.minimum(np.abs(grad_H_cons_lb), np.abs(grad_H_cons_ub))

ax[1, 1].plot(x[x_cons_valid], abs_grad_H_cons_lb[x_cons_valid],
         color=palette[2], linestyle='--')
ax[1, 1].plot(x[x_cons_valid], abs_grad_H_cons_ub[x_cons_valid],
         color=palette[2], label='Primal Feasible')
ax[1, 1].axvspan(a, b, color='black', alpha=0.1)

ax[1, 1].set_xlabel(r"$a$")
ax[1, 1].set_ylabel(r"$|\nabla\mathcal{H}|$")
ax[1, 1].set_xticks([0, 0.5, 1])

fig.tight_layout()
fig.align_ylabels()
fig.savefig('outputs/entropy_bounds.pdf', bbox_inches='tight')

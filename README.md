Repository for code and figures associated with the paper 'Online Neural Connectivity Estimation with Noisy Group Testing'.

## How can we get connectivity between large systems of neurons in vivo?

A naive approach (left figure) could stimulate each neuron, one at a time, and observe the responses of all other neurons. This would take an extremely long time for large networks, however. 
Using stimulations of small ensembles and a statistical method called group testing, we show in our recent paper (https://arxiv.org/abs/2007.13911) that this is now feasible even in networks of up to 10 thousand neurons: 10^8 connections! 
Check out the comparison of our method (right) below:

![](https://web.duke.edu/mind/level2/faculty/pearson/assets/videos/stim/Duke_logo.gif)

## Requirements
Our algorithm is implemented using only a few libraries: numpy, cupy (for GPU), and matplotlib (for plotting). We used python version 3.6.10 and conda to install these libraries. 
We provide an environment file for use with conda to create a new environment with these few requirements. The code has only been tested with the specified libraries and versions.
``` 
    conda env create --file environment.yml
    conda activate binarystim
```

##  Evaluation
Our code can be run with any of these methods as specified in the paper: batch, stream, adapt, naive, and exact. For example, to run the base case in batch mode, use: 
```
  python model_stim.py batch
```
As a helpful example, to generate a set of data for different number of tests and plot the results use:
```
  python model_stim.py batch_T
```

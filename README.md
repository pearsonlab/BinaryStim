Repository for code and figures associated with the paper 'Online Neural Connectivity Estimation with Noisy Group Testing'.

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

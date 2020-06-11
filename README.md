# BinaryStim
Repository for code and figures associated with the paper 'Online Neural Connectivity Estimation with Noisy Group Testing'

## Requirements
We recommend using conda to create a new environment with these dependencies. The code has only been tested with the specified libraries and versions.
``` 
    conda create --name binarystim --file requirements.txt
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

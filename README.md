# CARE batch

Functions for batch restoration with [CARE](https://csbdeep.bioimagecomputing.com/tools/care/).

## Requirements

- python 3.7

## Installation

**Option 1:**

1. Download the latest version of the package
2. cd into the package directory
3. Install the package by running `pip install .`
    
You can install the care_batch into your base python environment, but we recommend creating 
a new [anaconda](https://docs.anaconda.com/anaconda/install/) 
or [mamba](https://github.com/mamba-org/mamba) environment with python 3.7

**Option 2:**

Create a new conda environment with all dependencies from the provided yml file: 
   
`conda env create -f csbdeep.yml`

To overwrite an old environment with the same name, run the above command with the `--force` flag:

`conda env create -f csbdeep.yml --force`

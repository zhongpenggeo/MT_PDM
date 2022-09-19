## physics-driven model

### Requirements

- `PyTorch` >=1.8.0. Pytorch 1.8.0 starts to support complex numbers and it has a new implementation of FFT

- `torchinfo`

- `ray`  is used to generate smaples in arallel

- `numpy`

- `scipy`

  

### files
TE (mode xy) and TM (mode yx) are used to compute Ex and Hy, respectively.

There are four five subfolders in each mode. For example, in the TM:

- run: containing excutable python files
  - `conv_TM.py` is the main program for predicting Hx
  - `conv_TM_MSE.py` uses the mean square error (MSE) as loss function
  - `conv_TM_TFM.py` uses the TFM as governing equation (structure A1)
  - `conv_TM_SFM.py` uses the secondary field as output (structure A3)
  - `config.yml` is a configuration file written in yaml format


- utils: some auxiliary python fiels
  - `FNO.py` is the Fourier Neural Operator([Li et al., 2021](https://arxiv.org/abs/2010.08895))
  - `load_data.py` is used to load and normalize the input data
  - `derivative.py` computes the derivatives using finite difference.

- model:  saving trained model file (.pkl)
- Log: saving log file
- temp: if stoping program early, you can find model file (.pkl) here.

### Data
using Gaussian random field (GRF) generates conductivity structures, and computing the MT responses by finite difference method.

`non_grid-64-ray.ipynb` generates dataset in non-uniform grid

`grid-64-ray.ipynb` generates dataset in uniform grid


### Usage

```shell
cd ./run
# the last input is the item name in the config.yml
python conv_TE.py grid_5000
```


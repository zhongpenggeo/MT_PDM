## Introduction
The code is used to implement physics-driven model (PDM) and can be divided into three parts: data generation, model training, and model prediction.

## Data generation
We generate the conductivity structures using Gaussian random field (GRF), and compute the MT responses by finite difference method (FDM).

There are two files for generating conductivity structuress in uniform and non-uniform grids, respectively.

- `non_grid-64-ray.ipynb` generates dataset in non-uniform grid. The main parameters in the code are located in the second cell. You need to modify two parameters at least, the number of samples `n_sample` and the file name `file_name`. It is important to note that the saved training data and test data should be set with different file names.

- `grid-64-ray.ipynb` generates dataset in uniform grid. 

In addition, there are two library files, `gaussian_random_fields.py` is the modules of GRF, and `MT2D_secondary_direct.py` is the code for 2-D MT forwad modeling.


## Model training
directory `TE` and `TM` are used for xy- and yx-modes, respectively. The architecture of these two directorys is the same, so let's use `TE` as an example to illustrate.

- `run`: containing excutable python files
  - `conv_TM.py` is the main program for predicting Hx
  - `conv_TM_MSE.py` uses the mean square error (MSE) as loss function
  - `conv_TM_TFM.py` uses the TFM as governing equation
  - `conv_TM_SFM.py` uses the secondary field as output
  - `config.yml` is a configuration file written in yaml format. For the meaning of each parameter, refer to the comment on the corresponding line.


- `utils`: some auxiliary python files
  - `FNO.py` is the Fourier Neural Operator([Li et al., 2021](https://arxiv.org/abs/2010.08895))
  - `load_data.py` is used to load and normalize the input data
  - `derivative.py` computes the derivatives using finite difference.

- `model`:  saving trained model file (.pkl). (If the directory does not exist, you need to create it manually)
- `Log`: saving log file. (If the directory does not exist, you need to create it manually)
- `temp`: if stoping program early, you can find model file (.pkl) here. (If the directory does not exist, you need to create it manually)

## Model prediction
copy the saved models in `TE/run/model` and `TM/run/model` to the directory `eval/model` after the training is completed. Then run the `non_grid_plot.ipynb` to plot the predicted results.

## Usage
### data genetation
First, create the subdirectory  `data` under the directory `Data`. 
Second, Choose one of following two methods to generate conductivity sturctures and corresponding apparent resistivity and phase.

1. download data from from Google Cloud Drive in https://drive.google.com/drive/directorys/1RvBw3HU-hTbr6mWkF6qTyqscGDvM6FRH?usp=sharing.

or 

2. generate data locally: run `grid-64-ray.ipynb` or `non-grid-64-ray.ipynb` twice in Jupyter (each time you need to modify the parameter `file_name`) to consturct training dataset and test dataset.

Finally, check if there is `.mat` file under the directory `data`.

### model training
for xy-mode,
```shell
cd ./TE/run
# the last input is the item name in the config.yml
python conv_TE.py grid_5000
```

for yx-mode,
```shell
cd ./TM/run
# the last input is the item name in the config.yml
python conv_TM.py grid_5000
```

### model prediction (plot results)
1. change directory to `eval` and make new subdirectory `model`.
2. copy the saved model files from `TE/run/model` and `TM/run/model` to the directory `eval/model`
3. run the `non_grid_plot.ipynb` in Jupyter.


## Requirements

- `PyTorch` >=1.8.0. Pytorch 1.8.0 starts to support complex numbers and it has a new implementation of FFT

- `torchinfo`

- `ray`

- `numpy`

- `scipy`

- `matplotlib`

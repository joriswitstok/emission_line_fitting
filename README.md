# Emission-line fitting

## Contents
1. [Description](#Description)
2. [Installation and setup](#Installation)
    - [Cloning](#Cloning)
    - [Package requirements](#Package_requirements)
3. [Example usage](#Example_usage)

## <a name="Description"></a>Description

This is an emission-line fitting code using the pymultinest package (Feroz et al. 2009; Buchner et al. 2014) to fit and plot emission-line spectra. It will be described in [Witstok et al. (in prep.)](). Below, its usage is illustrated with an example.

## <a name="Installation"></a>Installation and setup

### <a name="Cloning"></a>Cloning

First, obtain and install the latest version of the code, which can be done via `pip`. You can clone the repository by navigating to your desired installation folder and using

```
git clone https://github.com/joriswitstok/emission_line_fitting.git
```

### <a name="Package_requirements"></a>Package requirements

Running the code requires the following Python packages:
- `numpy`
- `scipy`
- `astropy`
- `mpi4py`
- `pymultinest`
- `spectres`
- `corner`
- `matplotlib`
- `seaborn`
- `mock`
  
Most of these modules are easily installed via the file `elf3.yml` provided in the main folder, which can be used to create an `conda` environment in Python 3 that contains all the required packages (see the `conda` [documentation on environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details). However, the `emission_line_fitting` code and several MPI-related packages (`mpi4py` and `pymultinest`) should be installed via `pip` to ensure properly linking to a local MPI installation (if using MPI functionality).

If you have `conda` installed and set up as a `python` distribution, creating an environment can be achieved with:

```
conda env create -f elf3.yml
```

Before running the code, the environment needs to be activated using

```
conda activate elf3
```

By default, the terminal will indicate the environment is active by showing a prompt similar to:

```
(elf3) $ 
```

After navigating into the installation folder (see [Cloning](#Cloning)), the `emission_line_fitting` code is then installed into your `python` distribution via `pip` (`pip3`). NB: the `pip` executable related to the `conda` environment (or any other `python` distribution) should be used here - to verify which `pip` executable is active, use `which pip`. For example:

```
(elf3) $ which pip3
pip3 is /Users/Joris/anaconda3/envs/elf3/bin/pip3
(elf3) $ cd emission_line_fitting
(elf3) $ ls
LICENSE				elf3.yml			setup.py
README.md			emission_line_fitting
build				emission_line_fitting.egg-info
(elf3) $ pip3 install .
```

## <a name="Example_usage"></a>Example usage

A more advanced example usage case is illustrated by running the file `example_fit.py` (again located in the `examples` folder). This script performs a fitting routine to the observed spectrum of GS-z13 (as in [Hainline et al. 2024]()), given a resolution curve and an intrinsic model spectrum.
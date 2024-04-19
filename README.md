# Train Characteristics Detector
***

This project uses as input Waterfall data obtained from a Distributed Acoustic Sensing (DAS) device installed in a train track
section between the Renfe link of Málaga - Córdoba in Andalucia, Spain. 

Using this data, is able to compute many rail and train characteristics such as train-speed, the track (rail) which
the train is passing through, direction and also classifies the train by comparing a DTW distance of the 
"train footprints" computed with a pre-defined base values of each type of train.

## Getting started
***

### Prerequisites

The tool has been tested on python 3.9.11.

#### Python Installation (Windows)

1. Download [Windows installer (64-bit)](https://www.python.org/ftp/python/3.9.11/python-3.9.11-amd64.exe) python installer for Windows from [Official Python distribution website](https://www.python.org/downloads/windows/)
2. Follow installation instructions
3. Create new Virtual environment called `venv` inside project root path.

```shell
py -3.9 -m venv "C:\[project-root-path]\venv"
```


#### Python Installation (Linux) (Tested on Ubuntu 18.04 Bionic)

We will explore two python installation methods:

##### Installing Python from dpkg package manager

First ensure you have updated and upgraded linux to the latest version
```shell
sudo apt update && sudo apt upgrade
```

Check already installed python version
```shell
python3 --version
python --version
```

Install python version from dpkg package manager
```shell
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.9
```

NOTE: This method has not worked in IPV server machine on Ubuntu 18.04 Bionic release

##### Manual Installation

These steps explain how to install and compile Python specific version from source and then create a new python virtual 
environment `venv`. 

First Install `virtualenv`

```shell
sudo apt install virtualenv
```

```shell
pip install virtualenv
```

1. Create a new directory where you want to download Python distribution (Ex. `/usr/src`)

Ex.
```shell
mkdir ~/src
```

2. Get Official Python distribution
```shell
cd ~/src
wget https://www.python.org/ftp/python/3.9.11/Python-3.9.11.tgz
```

3. Unzip
```shell
tar -zxvf Python-3.9.11.tgz
```

4. Create a new directory where compile python software 
```shell
mkdir ~/.python3.9.11
```

5. Configure Python makefile
```shell
cd Python-3.9.11
./configure --prefix=$hOME/.python3.9.11
```

6. Make and Install
```shell
make
make install 
```

7. Run setup.py using recently install python version
```shell
~/.python3.9.11/bin/python3 setup.py install
```

8. Create virtual environment

#### Fixing possible errors

You can face the following error when building Python from source:

```shell
Failed to build these modules:
_hashlib              _ssl   
etc...                                  

Could not build the ssl module!
Python requires an OpenSSL 1.0.2 or 1.1 compatible libssl with X509_VERIFY_PARAM_set1_host().
LibreSSL 2.6.4 and earlier do not provide the necessary APIs, https://github.com/libressl-portable/portable/issues/381
```

This [stackoverflow question](https://stackoverflow.com/questions/53543477/building-python-3-7-1-ssl-module-failed) and
this [blog](https://jameskiefer.com/posts/installing-python-3.7-on-debian-8/) explores the different solutions you may have.

Solution found.

Install required packages:
```shell
sudo apt-get install build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
```

Download oppenssl (1.0.2o version at least) and point to that specific version.
```shell
./configure --with-openssl=../openssl-1.0.2o --enable-optimizations --prefix=$HOME/.python3.9.11
sudo make
sudo make altinstall
```

### Installation (tested on Python 3.9.11)

From project root path:

1. Activate `venv` virtual environment

On Windows:
```shell
venv\Scripts\activate
```

On Linux:
```shell
source venv/bin/activate
```

2. Upgrade pip
```shell
python -m pip install --upgrade pip
```

3. Install requirements
```shell
pip install -r requirements.txt 
```

## Data
***

### Data Prerequisites

First of all, you will need to store the data in a specific path. By default, in a development environment,
the path is defined using the following path schema:

```
..data/{project_name}/{file_extension}/{year}/{month}/{day}
```

You can edit this configuration values in the `Config Paramenters` section located in `main.py`
You can also set `PRODUCTION_ENVIRONMENT` flag to `True` and define your custom absolute data path.


> In the defined absolute path you should manually create `{year}/{month}/{day}` folder structure
and store there waterfall data.

### Data convection

The waterfall data files are named with the hour, minute and second in which an event has been detected in the DAS 
equipment installed and monitoring railway in real-time, using the format `{hour}_{minute}_{second}` 
(Ex. `10_57_45.json`). 

The data is stored in JSON by default (`.json`)(following client wishes), but also admits numpy array data `npy` data,
which obtains ~80 times faster loading times.

You will need to set in the `Configuration Paramenters` the `file_extension` parameter to `npy` or `json`

#### Base data convection

If you want to classify between different train classes, a `base-data` of each different `train-class` MUST be defined.

The `base-data` contains the computed `train-track` (the footprint of the train) of a speficic `train-class`

The `base-data` will be stored in the path: `../data/{project_name}/{file_extension}/base`

There we should manually create `train-type` directories with the same name of the ones defined in `train_map` variable:

Ex. `../data/{project_name}/{file_extension}/base/S-102-6p`

Defined `train_map`

```python
train_map = {
    1: "S-102-6p",
    2: "S-102-8p",
    3: "S-103-u",
    4: "S-103-d",
    5: "S-104"
}
```

### Train Characteristic Schema 

It has been defined the following schema:

```python
train_char_schema = {
    "datetime": None,
    "event": None,
    "status": "not-computed",
    "direction": None,
    "speed": None,
    "speed-magnitude": "kmh",
    "speed-error": None,
    "rail-id": None,
    "rail-id-confidence": None,
    "train-id": None,
    "train-id-confidence": None,
    "train-ids": train_ids,
}
```

## Usage
***

Compute train characteristics of a specific waterfall data and show results. 

```shell
python .\main.py -f 07_06_27.npy -d
```

By default, the input file will be searched for in the path  `{data_path}/{year}/{month}/{day}`, defined in 
`Configuration Paramenters`. You can also set the `{year}/{month}/{day}` from command line.

```shell
 python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d 
```

Expected result:
```shell
INFO:__main__:CLASSIFY_TRAINS: True
DEBUG:__main__:train-characteristic-values:
{
  'datetime': None,
  'event': 'train',
  'status': 'computed',
  'direction': 'Malaga -> Cordoba',
  'speed': 290.005, 'speed-magnitude': 'kmh',
  'speed-error': 6.252,
  'rail-id': 1,
  'rail-id-confidence': 0.488,
  'train-id': 1,
  'train-id-confidence': 0.997,
  'train-ids': [1, 2, 3, 4, 5],
  'train-class': 'Pending to be verified'
}
```

You can add the results and store them in the give JSON file using the flag `--serialize` or `-s`. A JSON file will be created in the path
```
..data/{project_name}/{file_extension}/output/{year}/{month}/{day}
```

```shell
python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -s
```
> WARNING: This option will only add data to JSON file if the input data was a JSON file in the first time.
> 
> The computed train characteristics will be added in the default key ´info´

#### Multi-regression (NEW!)

You can also compute the `multi-regression` method, to improve the mask detection accuracy and obtain a better 
fitted `rail-view`. 

```shell
python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -mr 1
```

> INFO:
> 
>  - Use '-mr' option with a value equal or higher than '1' 
> 
>  - 'multi-regression-max-epochs': is set to 4 by default. So you will not be able to compute the regression more than 
> this number of iterations
> 
>  - 'multi-regression-mask-width-margin' sets the percentage in which the mask-width will be reduced. Ex
> if 0.1 -> mask-width = mask-width * (1 -  0.1). So, every iteration mask-width will be reduced by a 10%
> Very High or low values will result in a bad functioning of the method. Be careful tuning this variable.

### Plot Process Waterfall's

The plotting of the waterfall in each different process will help to debug possible data related problems in the future


#### Show Raw Waterfall

![raw_w_sample_01.png](img%2Fraw_w_sample_01.png)

```shell
 python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -sec 0 -rw
```

> You can select the section to be showed usin `-sec` option. Available sections '0' and '1'.

#### Show Filtered Waterfall

![filtered_w_sample_01.png](img%2Ffiltered_w_sample_01.png)

```shell
 python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -sec 0 -fw
```

#### Show Sobel Waterfall

![sobel_w_sample_01.png](img%2Fsobel_w_sample_01.png)

```shell
 python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -sec 0 -sw
```

#### Show Threshold Waterfall (one-hot-encoding)

![thr_w_sample_01.png](img%2Fthr_w_sample_01.png)

```shell
 python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -sec 0 -tw
```

#### Show Mean Filtered Waterfall (one-hot-encoding)

![thr_mean_w_sample_01.png](img%2Fthr_mean_w_sample_01.png)

```shell
 python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -sec 0 -mfw
```

#### Show Computed Mask Waterfall (one-hot-encoding)

![mask_w_sample_01.png](img%2Fmask_w_sample_01.png)

```shell
 python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -sec 0 -mw
```

#### Show Computed Rail View

![rail_view_w_sample_01.png](img%2Frail_view_w_sample_01.png)

```shell
 python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -sec 0 -rv
```

#### Show Train Track (train's footprint)

![train_track_w_sample_01.png](img%2Ftrain_track_w_sample_01.png)

```shell
 python .\main.py -dt 2023-03-09 -f 15_26_23.npy -d -sec 0 -tt
```

### Execute main from within a second script

This will be helpful to automate tasks in production code, using the already implemented features.

Ex. Execute train characteristics for previous data sets that had not been calculated

To import the argparse from external script, you will need to:

1) Append path to sys.path and import main function

```python
import sys
from pathlib import Path

file_path = Path(__file__).resolve()
parent_path = file_path.parent

sys.path.append(str(parent_path))

from main import main
```

2) Run main passing each argument as a string in a list

```python
main(["-dt", "2023-03-09", "-f", "15_26_23.npy", "-d"])
```

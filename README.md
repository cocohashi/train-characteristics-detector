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

```bash
py -3.9 -m venv "C:\[project-root-path]\venv"
```


#### Python Installation (Linux) (Tested on Ubuntu 18.04 Bionic)

We will explore two python installation methods:

##### Installing Python from dpkg package manager

First ensure you have updated and upgraded linux to the latest version
```bash
sudo apt update && sudo apt upgrade
```

Check already installed python version
```bash
python3 --version
python --version
```

Install python version from dpkg package manager
```bash
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

```bash
sudo apt install virtualenv
```

```bash
pip install virtualenv
```

1. Create a new directory where you want to download Python distribution (Ex. `/usr/src`)

Ex.
```bash
mkdir ~/src
```

2. Get Official Python distribution
```bash
cd ~/src
wget https://www.python.org/ftp/python/3.9.11/Python-3.9.11.tgz
```

3. Unzip
```bash
tar -zxvf Python-3.9.11.tgz
```

4. Create a new directory where compile python software 
```bash
mkdir ~/.python3.9.11
```

5. Configure Python makefile
```bash
cd Python-3.9.11
./configure --prefix=$hOME/.python3.9.11
```

6. Make and Install
```bash
make
make install 
```

7. Run setup.py using recently install python version
```bash
~/.python3.9.11/bin/python3 setup.py install
```

8. Create virtual environment

#### Error fixes

You can face the following error when building Python from source:

```bash
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
```bash
sudo apt-get install build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
```

Download oppenssl (1.0.2o version at least) and point to that specific version.
```bash
./configure --with-openssl=../openssl-1.0.2o --enable-optimizations --prefix=$HOME/.python3.9.11
sudo make
sudo make altinstall
```

### Installation (tested on Python 3.9.11)

From project root path:

1. Activate `venv` virtual environment

On Windows:
```bash
venv\Scripts\activate
```

On Linux:
```bash
source venv/bin/activate
```

2. Upgrade pip
```bash
python -m pip install --upgrade pip
```

3. Install requirements
```bash
pip install -r requirements.txt 
```

## Usage
***


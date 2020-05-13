# SABer

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1a2954edef114b81a583bb23ffba2ace)](https://app.codacy.com/gh/hallamlab/SABer?utm_source=github.com&utm_medium=referral&utm_content=hallamlab/SABer&utm_campaign=Badge_Grade_Dashboard)

SAG Anchored Binner for recruiting metagenomic reads using single-cell amplified genomes as references

Check out the [wiki](https://github.com/hallamlab/SABer/wiki) for tutorials and more information on SABer!!

## WARNING: SABer has only been tested on Linux and Python 3.5 or greater
## Also, SABer is under HEAVY development right now, so the code and documentation is very dynamic
## Here are the install instructions for SABer (until its on PyPI)
Currently the easiest way to install SABer is to use a conda virtual environment.  
This will require the installation of [Anaconda](https://www.anaconda.com/distribution/).  
Once Anaconda is installed, you can follow the directions below to install all dependencies  
and SABer within a conda environment.
```sh
git clone --recurse-submodules git@github.com:hallamlab/SABer.git  
cd SABer  
```
### Create conda env
```sh
conda create -n saber_env python=3.8  
conda activate saber_env  
```
### Install BWA
```sh
conda install -c bioconda bwa
```
### Install python dependencies
```sh
pip install -r requirements.txt  
```
### Special install for SamSum for now
```sh
cd samsum  
python3 setup.py sdist  
pip install dist/samsum*tar.gz  
cd ..  
```  
### Install SABer
```sh
python3 setup.py sdist bdist_wheel  
pip install dist/SABerML-0.0.1-py3-none-any.whl  
```

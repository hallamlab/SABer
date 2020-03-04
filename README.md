# SABer
SAG Anchored Binner for recruiting metagenomic reads using single-cell amplified genomes as references

Check out the [wiki](https://github.com/hallamlab/SABer/wiki) for tutorials and more information on SABer!!


## Here are the install instructions for SABer (until its on PyPI):
```
git clone --recurse-submodules git@github.com:hallamlab/SABer.git  
cd SABer  
```
### Create conda env:
```
conda create -n saber_env python=3.5  
conda activate saber_env  
```
### Install BWA
```
conda install -c bioconda bwa
```
### Install python dependencies:
```
pip install -r requirements.txt  
```
### Special install for SamSum for now:
```
cd samsum  
python3 setup.py sdist  
pip install dist/samsum*tar.gz  
cd ..  
```  
### Install SABer:
```
python3 setup.py sdist bdist_wheel  
pip install dist/SABer-0.0.1-py3-none-any.whl  
```

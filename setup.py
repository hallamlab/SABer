from setuptools import setup

with open("README.md", "r") as readme:
    LONG_DESCRIPTION = readme.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X",
    "Programming Language :: Python :: 3.5",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

pks = ['saber']

SETUP_METADATA = \
    {
        "name": "SABerML",
        "version": "0.0.1",
        "description": "Software for recruiting metagenomic reads using single-cell amplified genome data.",
        "long_description": LONG_DESCRIPTION,
        "long_description_content_type": "text/markdown",
        "author": "Ryan McLaughlin, Connor Morgan-Lang",
        "author_email": "mclaughlinr2@gmail.com",
        "url": "https://github.com/hallamlab/SABer",
        "license": "GPL-3.0",
        "include_package_data": True,
        "package_dir": {'': 'src'},  # Necessary for proper importing
        "packages": pks,
        "package_data": {
            'saber': ['configs/*']},
        "entry_points": {'console_scripts': ['saber = saber.__main__:main']},
        "classifiers": CLASSIFIERS,
        "install_requires":
            ['attrs==22.1.0', 'backcall==0.2.0', 'boltons==21.0.0',
             'bz2file==0.98', 'CacheControl==0.12.10',
             'cachetools==4.2.4', 'certifi==2021.10.8',
             'cffi==1.15.0', 'charset-normalizer==2.0.7',
             'contextlib2==21.6.0', 'cycler==0.11.0',
             'Cython==0.29.24', 'debtcollector==2.3.0',
             'decorator==5.1.0', 'deprecation==2.1.0',
             'dit==1.2.3', 'fonttools==4.28.1', 'hdbscan==0.8.27',
             'hdmedians==0.14.2', 'idna==3.3',
             'importlib-metadata==4.8.2', 'iniconfig==1.1.1',
             'ipython==7.29.0', 'jedi==0.18.1', 'joblib==1.1.0',
             'kiwisolver==1.3.2', 'llvmlite==0.37.0',
             'lockfile==0.12.2', 'matplotlib==3.5.0',
             'matplotlib-inline==0.1.3', 'msgpack==1.0.2',
             'natsort==8.0.0', 'networkx==2.6.3',
             'numba==0.54.1', 'packaging==21.3',
             'pandas==1.3.4', 'parso==0.8.2', 'pbr==5.8.0',
             'pexpect==4.8.0', 'pickleshare==0.7.5',
             'Pillow==9.0.1', 'pluggy==1.0.0', 'prettytable==2.4.0',
             'prompt-toolkit==3.0.22', 'ptyprocess==0.7.0',
             'py==1.11.0', 'pycparser==2.21', 'pyfastx==0.8.4',
             'Pygments==2.10.0', 'pynndescent==0.5.5',
             'pyparsing==3.0.6', 'pytest==7.1.3',
             'python-dateutil==2.8.2', 'pytz==2021.3',
             'requests==2.26.0', 'scikit-bio==0.5.6',
             'scikit-learn==1.0.1', 'scipy==1.7.2', 'screed==1.0.5',
             'setuptools-scm==6.3.2', 'six==1.16.0',
             'sourmash==4.2.2', 'threadpoolctl==3.0.0',
             'tomli==1.2.2', 'tqdm==4.62.3', 'traitlets==5.1.1',
             'typing_extensions==4.0.0', 'umap-learn==0.5.2',
             'urllib3==1.26.7', 'wcwidth==0.2.5', 'wrapt==1.13.3',
             'zipp==3.6.0'
             ]
    }

setup(**SETUP_METADATA)

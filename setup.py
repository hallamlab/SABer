from setuptools import setup, find_packages

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
        "packages": find_packages('src', exclude=["tests"]),
        "include_package_data": True,
        "package_dir": {'saber': 'src/saber'},  # Necessary for proper importing
        #"package_data": {'tests': ["tests/test-data/*.sam"]},
        #"py_modules": [splitext(basename(path))[0] for path in glob('src/*.py')],
        "entry_points": {'console_scripts': ['saber = saber.__main__:main']},
        "classifiers": CLASSIFIERS,
        "install_requires": ["numpy", "pytest"]
    }

setup(**SETUP_METADATA)

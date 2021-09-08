ocean_data_gateway
==============================
[![Build Status](https://img.shields.io/github/workflow/status/axiom-data-science/ocean_data_gateway/Tests?logo=github&style=for-the-badge)](https://github.com/axiom-data-science/ocean_data_gateway/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/axiom-data-science/ocean_data_gateway.svg?style=for-the-badge)](https://codecov.io/gh/axiom-data-science/ocean_data_gateway)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/ocean_data_gateway/latest.svg?style=for-the-badge)](https://ocean_data_gateway.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/workflow/status/axiom-data-science/ocean_data_gateway/linting%20with%20pre-commit?label=Code%20Style&style=for-the-badge)](https://github.com/axiom-data-science/ocean_data_gateway/actions)
[![Conda Version](https://img.shields.io/badge/Anaconda.org-V0.7.0-blue.svg?style=for-the-badge)](https://anaconda.org/conda-forge/ocean_data_gateway)
[![Python Package Index](https://img.shields.io/pypi/v/ocean_data_gateway.svg?style=for-the-badge)](https://pypi.org/project/ocean_data_gateway)

Your gateway to ocean data.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>


## Installation

Install the package plus its requirements from PyPI with
``` bash
$ pip install ocean_data_gateway
```

or from `conda-forge` with
``` bash
$ conda install -c conda-forge ocean_data_gateway
```

Clone the repo:
``` bash
$ git clone https://github.com/axiom-data-science/ocean_data_gateway.git
```

In the `ocean_data_gateway` directory, install conda environment:
``` bash
$ conda env create -f environment.yml
```

For local package install, in the `ocean_data_gateway` directory:
``` bash
$ pip install -e .
```

To also develop this package, install additional packages with:
``` bash
$ conda install --file requirements-dev.txt
```

To then check code before committing and pushing it to github, locally run
``` bash
$ pre-commit run --all-files
```

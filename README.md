ocean_data_gateway
==============================
[![Build Status](https://github.com/axiom-data-science/ocean_data_gateway/workflows/Tests/badge.svg)](https://github.com/axiom-data-science/ocean_data_gateway/actions)
[![codecov](https://codecov.io/gh/axiom-data-science/ocean_data_gateway/branch/master/graph/badge.svg)](https://codecov.io/gh/axiom-data-science/ocean_data_gateway)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)[![pypi](https://img.shields.io/pypi/v/ocean_data_gateway.svg)](https://pypi.org/project/ocean_data_gateway)
<!-- [![conda-forge](https://img.shields.io/conda/dn/conda-forge/ocean_data_gateway?label=conda-forge)](https://anaconda.org/conda-forge/ocean_data_gateway) -->[![Documentation Status](https://readthedocs.org/projects/ocean_data_gateway/badge/?version=latest)](https://ocean_data_gateway.readthedocs.io/en/latest/?badge=latest)


Your gateway to ocean data.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>


## Installation

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

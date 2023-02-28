# Scikit-ReducedModel

<img src="logo.png" alt="logo" width="60%">

[![PyPI version](https://badge.fury.io/py/Scikit-ReducedModel.svg)](https://badge.fury.io/py/Scikit-ReducedModel)
[![Python version](https://img.shields.io/badge/python-3.10%20-blue)](https://img.shields.io/badge/python-3.10%20-blue)
[![Documentation Status](https://readthedocs.org/projects/scikit-reducedmodel/badge/?version=latest)](https://scikit-reducedmodel.readthedocs.io/en/latest/?badge=latest)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)
[![Scikit-ReducedModel CI](https://github.com/francocerino/scikit-reducedmodel/actions/workflows/ci.yml/badge.svg)](https://github.com/francocerino/francocerino/actions/workflows/ci.yml)

[![Code Coverage](https://img.shields.io/codecov/c/github/francocerino/scikit-reducedmodel)](https://codecov.io/github/francocerino/scikit-reducedmodel)
![GitHub](https://img.shields.io/github/license/francocerino/scikit-reducedmodel)
![Depfu](https://img.shields.io/depfu/francocerino/scikit-reducedmodel)
[![Black Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Scikit-ReduceModel is a Python package to build [reduced order models](https://en.wikipedia.org/wiki/Model_order_reduction). This code gives the
standard reduced-basis framework and provides an efficient and accurate solution for model building. Also, an gives an extension of it, implementing the hp-greedy refinement strategy, an enhancement approach for reduced-basis model building. This approach uses an automatic parameter space partitioning, where there is a local reduced basis on each partition. The procedure splits spaces in two, in a recursive way.

Surrogate models can be built, which are comprised of three sequential steps of data processing that extract the most relevant information and use it to learn the patterns of the solutions and build easy to evaluate expressions: building a reduced basis, applying the empirical interpolation method and using a machine learning algorithm to learn the behavior of the solutions to obtain the surrogate model.

This package is built with the philosophy and idea of usability of scikit-learn modules. For usage examples, see the [documentation](https://scikit-reducedmodel.readthedocs.io/en/latest/).

# Motivation

In science and engineering is known that obtaining numerical simulations by solving differential equations can be more computational demanding than desired. For example, in the field of general relativity, to obtain expressions of gravitational waves could cost months using supercomputers. Furthermore, there are studies as parameter estimation that can require up to millions of sequential estimations, dominating the computational expense of the problem. In the last years, these problems were addresed building surrogate models from high cost simulations, taking advantage of the redundancy of the solutions with respect to the parameter space, which can build solutions in real time with no loss of accuracy.

# Installation

To install the latest stable version of ScikitReducedModel from PyPI:

```bash
pip install skreducedmodel
```

To install the developer version (may be unstable):

```bash
git clone https://github.com/francocerino/scikit-reducedmodel
cd scikit-reducedmodel
pip install .
```

# Quick Usage

In order to build a surrogate model, we need to be familiar with a set of functions parameterized by $λ$, denoted as $f_λ(x)$.

The known functions at given parameters are named as the training set (`training_set`).

The associated parameters to `training_set` are `parameters`.

We need also a distretization of $x$ `x_set`.

Then, we can first build the reduced basis, in this case, we use the default parameters.

```python

from skreducedmodel.reducedbasis import ReducedBasis

rb = ReducedBasis()
rb.fit(training_set = training_set,
       parameters = parameters
       physical_points = x_set)
```

In the second step, with the reduced basis built, the empirical interpolation method is applied.

```python

from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

eim = EmpiricalInterpolation(rb)
eim.fit()
```

Finally, we build the reduced model from our eim object

```python

from skreducedmodel.surrogate import Surrogate

model = Surrogate(eim)
model.fit()
```

In case we are interested in studying only `Surrogate` objects, the package has a function that automates the whole process.

```python

from skreducedmodel.mksurrogate import mksurrogate

surrogate = mksurrogate(parameters = param,
                        training_set = training_set,
                        physical_points = x_set,
                        )

```

## Contributions

We encourage users to contribute with ideas, code, or by reporting bugs. To report bugs or issues, users should create an issue in the project repository. To contribute with code, please submit a pull request. We suggest that you contact us at francocerino@gmail.com prior to undertaking any significant improvement that requires substantial effort to address technical and design aspects before beginning development.

## Authors

- Franco Cerino <[francocerino@gmail.com](francocerino@gmail.com)> ([FaMAF-UNC][]).

- Agustín Rodríguez-Medrano ([IATE-OAC-CONICET][], [FaMAF-UNC][]).

[FaMAF-UNC]: https://www.famaf.unc.edu.ar/
[IATE-OAC-CONICET]: http://iate.oac.uncor.edu/

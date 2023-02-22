# Scikit-ReducedModel

<img src="logo.png" alt="logo" width="60%">

[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)
[![Python version](https://img.shields.io/badge/python-3.10%20-blue)](https://img.shields.io/badge/python-3.10%20-blue)
[![Documentation Status](https://readthedocs.org/projects/scikit-reducedmodel/badge/?version=latest)](https://scikit-reducedmodel.readthedocs.io/en/latest/?badge=latest)

[![Scikit-ReducedModel CI](https://github.com/francocerino/scikit-reducedmodel/actions/workflows/ci.yml/badge.svg)](https://github.com/francocerino/francocerino/actions/workflows/ci.yml)



Scikit-ReduceModel is a Python package to construct [reduced models](https://en.wikipedia.org/wiki/Model_order_reduction) <>. This code is an extension of the
standard reduced-base framework and provides an efficient and accurate solution for model building.
It implements the hp-greedy refinement strategy, an enhancement approach for reduced-base model
building. The approach uses a parameter space partitioning, a local reduced basis and a binary tree
as the resulting structure, all obtained automatically.
The usability of this package is similar to that of the scikit-learn modules. For usage examples, see the documentation. 

# Installation

To install the latest stable version of ScikitReducedModel from PyPI:

```bash
pip install skreducedmodel
```

To install the developer version (may be unstable):

```bash
git clone https://github.com/francocerino/scikit-reducedmodel
cd scikir-reducedmodel
pip install .
```

# Quick Usage

In order to construct a reduced model, we require knowledge of a training set (training_set). 
That is, we need to be familiar with a set of functions parameterized 
by a real number λ, denoted as :math:`f_λ(x)`.

We need also a distretization of the :math:`x` (x_set) and of the :math:`λ` space (param).

Then, we can first built the reduced basis, in this case, we use the default parameters. 
```python

from skreducedmodel.reducedbasis import ReducedBasis

rb = ReducedBasis()
rb.fit(training_set = training_set,
       parameters = param
       physical_points = x_set)
```
The second step is built the empirical interpolator with the reduced basis generated
```python 

from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

eim = EmpiricalInterpolation(rb)
eim.fit()
```
Finally, we construct the reduced model from our eim object
```python 
   
from skreducedmodel.surrogate import Surrogate

model = Surrogate(eim)
model.fit()
```

In case we are interested in studying the ReducedBasis and EmpiricalInterpolation objects, the package has a function that automates the whole process.
```python

from skreducedmodel.mksurrogate import mksurrogate

surrogate = mksurrogate(parameters = param,
                        training_set = training_set,
                        physical_points = x_set,
                        )

```


## Authors

- Franco Cerino <[francocerino@gmail.com](francocerino@gmail.com)> ([FaMAF-UNC][]).

- Agustín Rodríguez-Medrano  ([IATE-OAC-CONICET][], [FaMAF-UNC][]).

[FaMAF-UNC]: https://www.famaf.unc.edu.ar/
[IATE-OAC-CONICET]: http://iate.oac.uncor.edu/

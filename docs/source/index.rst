.. ScikitReduceModel documentation master file, created by
   sphinx-quickstart on Thu Nov 17 10:47:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Scikit-ReduceModel's documentation!
===================================

.. image:: logo.png
   :width: 600
   :alt: scikit logo


`Scikit-ReduceModel <https://github.com/francocerino/scikit-reducedmodel>`_ is a Python package to construct reduced model.
This code is an extension of the standard reduced-base framework and provides an efficient 
and accurate solution for model building. It implements the `hp-greedy refinement <https://arxiv.org/abs/2212.08554>`_ strategy, an enhancement 
approach for reduced-base model building. The approach uses a parameter space partitioning, a local 
reduced basis and a `binary tree <https://en.wikipedia.org/wiki/Binary_tree>` as the resulting structure, all obtained automatically.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials/index.rst
   package
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Usage
===========

In order to construct a reduced model, we require knowledge of a training set (training_set). 
That is, we need to be familiar with a set of functions parameterized 
by a real number λ, denoted as :math:`f_λ(x)`.

We need also a distretization of the :math:`x` (x_set) and of the :math:`λ` space (param).

Then, we can first built the reduced basis, in this case, we use the default parameters. 
::

      from skreducedmodel.reducedbasis import ReducedBasis

      rb = ReducedBasis()
      rb.fit(training_set = training_set,
            parameters = param
            physical_points = x_set)

The second step is built the empirical interpolator with the reduced basis generated
:: 

      from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

      eim = EmpiricalInterpolation(rb)
      eim.fit()

Finally, we construct the reduced model from our eim object
:: 
   
      from skreducedmodel.surrogate import Surrogate

      model = Surrogate(eim)
      model.fit()

In case we are interested in studying the ReducedBasis and EmpiricalInterpolation objects, the package has a function that automates the whole process.
::
   
      from skreducedmodel.mksurrogate import mksurrogate

      surrogate = mksurrogate(parameters = param,
                              training_set = training_set,
                              physical_points = x_set,
                              )









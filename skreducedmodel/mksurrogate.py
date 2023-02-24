"""mksurrogate function implementation."""

import inspect

from .empiricalinterpolation import EmpiricalInterpolation
from .reducedbasis import ReducedBasis
from .surrogate import Surrogate


class InputError(ValueError):
    """Class used to give input errors of the function mksurrogate."""

    pass


def mksurrogate(instance=None, **kwargs):
    """Trains and returns a Surrogate model.

    Factory function for the classes ReducedBasis, EmpiricalInterpolation
    and Surrogate.
    In one way, only hyperparameters and training data can be given.
    In another way, can be given an instance of ReducedBasis or
    EmpiricalInterpolation and hyperparameters of the subsequent classes.
    If the instance will be trained if that did not happen before.

    Parameters
    ----------
    instance : _type_, optional
        reduced basis or eim instance, by default None

    kwargs:
    Training data or hyperparameters for the given instance.

    Returns
    -------
    Surrogate
        A trained surrogate model, ready to make predictions.

    Raises
    ------
    InputError
        Bad parameters given as input to the function.
        Cases:
            - A reduced basis or eim instance is not needed as input
            if training data is given. If want to specify
            hyperparameters when using a function or method, they
            must be passed as kwargs.
            - 'input' must not be given. It is not a parameter to use.
            - If an instance of ReducedBasis is given, must not
            be given hyperparameters of it.
            - There is no training data to build a surrogate model
            - if 'instance' is given, must be an instance of
            EmpiricalInterpolation or ReducedBasis.
    """
    # obtain given input data to train ReducedBasis, if needed.
    rb_fit_parameters = ["training_set", "parameters", "physical_points"]
    kwargs_rb_fit = {k: v for k, v in kwargs.items() if k in rb_fit_parameters}

    if kwargs_rb_fit != {} and instance is not None:
        raise InputError(
            (
                "A reduced basis or eim instance is not needed as input "
                + "if training data is given. In case hyperparameters "
                + "want to be specified, they must be kwargs."
            )
        )

    # obtain given input data to instantiate ReducedBasis or
    # EmpiricalInterpolation or Surrogate, if needed.
    parameters_rb = inspect.signature(ReducedBasis.__init__).parameters
    kwargs_rb = {k: v for k, v in kwargs.items() if k in parameters_rb}

    parameters_eim = inspect.signature(
        EmpiricalInterpolation.__init__
    ).parameters
    kwargs_eim = {k: v for k, v in kwargs.items() if k in parameters_eim}

    parameters_rom = inspect.signature(Surrogate.__init__).parameters
    kwargs_rom = {k: v for k, v in kwargs.items() if k in parameters_rom}

    # raise an error if there are variables in kwargs that will not be used.
    for key in kwargs.keys():
        if (
            key
            not in list(parameters_rb)[1:]  # do not include self.
            + rb_fit_parameters
            + list(parameters_eim)[2:]  # do not include self and rb.
            + list(parameters_rom)[2:]  # do not include self and eim.
        ):
            raise InputError(
                f"{key} must not be given. It is not a parameter to use."
            )

    # build the Surrogate model.
    if instance is None:
        # build it from the beggining
        rb = ReducedBasis(**kwargs_rb)
        rb.fit(**kwargs_rb_fit) if not rb.is_trained else None
        eim = EmpiricalInterpolation(rb, **kwargs_eim)
        eim.fit() if not eim.is_trained else None

    elif isinstance(instance, ReducedBasis):
        if kwargs_rb != {}:
            raise InputError(
                "If an instance of ReducedBasis is given, must not "
                + "be given hyperparameters of it."
            )

        # build it using the EmpiricalInterpolation instance given.
        instance.fit(**kwargs_rb_fit) if not instance.is_trained else None
        eim = EmpiricalInterpolation(instance, **kwargs_eim)
        eim.fit() if not eim.is_trained else None

    elif isinstance(instance, EmpiricalInterpolation):
        if not instance.reduced_basis.is_trained:
            raise InputError("There is no training data")

        instance.fit() if not instance.is_trained else None
        eim = instance
    else:
        raise InputError(
            "if 'instance' is given, must be an instance of "
            + "EmpiricalInterpolation or ReducedBasis."
        )

    surrogate = Surrogate(eim, **kwargs_rom)
    surrogate.fit()
    return surrogate

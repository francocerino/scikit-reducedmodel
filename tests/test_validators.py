import pytest
from skreducedmodel import validators


def test_validators():
    # assert validate_parameters(np.array([1, 2, 3]))
    with pytest.raises(TypeError):
        validators.validate_parameters([1, 2, 3])
    with pytest.raises(TypeError):
        validators.validate_physical_points([1, 2, 3])
    with pytest.raises(TypeError):
        validators.validate_training_set([1, 2, 3])

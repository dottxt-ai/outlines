import pytest

from outlines.models.base import Model


@pytest.fixture
def model_implementation():
    class ModelImplementation(Model):
        def generate(self, *args, **kwargs):
            pass

        def generate_stream(self, *args, **kwargs):
            pass

    return ModelImplementation


def test_model_init_no_tensor_library_name(model_implementation):
    model = model_implementation()
    with pytest.raises(NotImplementedError):
        model.tensor_library_name


def test_model_init_with_tensor_library_name(model_implementation):
    model = model_implementation(tensor_library_name="torch")
    assert model.tensor_library_name == "torch"

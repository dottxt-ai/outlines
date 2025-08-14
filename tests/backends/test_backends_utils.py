import torch
import numpy as np


def simulate_model_calling_processor(processor, tensor_library_name, vocabulary_size, eos_token_id, batch_size):
    if tensor_library_name == "torch":
        tensor_adapter = TorchTensorAdapter()
    elif tensor_library_name == "numpy":
        tensor_adapter = NumpyTensorAdapter()
    elif tensor_library_name == "mlx":
        tensor_adapter = MLXTensorAdapter()

    processor.reset()
    i = 0
    input_ids = tensor_adapter.randint(0, vocabulary_size, (batch_size, 10))
    while True:
        i += 1
        logits = tensor_adapter.randn((batch_size, vocabulary_size))
        output = processor(input_ids, logits)
        assert output.shape == (batch_size, vocabulary_size)
        if all(input_ids[:, -1] == eos_token_id):
            break
        input_ids = tensor_adapter.add_token_inputs_ids(input_ids, output)
        print(input_ids)
        if i > 20:
            break
    return input_ids[:, 10:]

class TorchTensorAdapter():
    def randn(self, shape):
        return torch.randn(*shape)

    def randint(self, low, high, size):
        return torch.randint(low, high, size)

    def add_token_inputs_ids(self, input_ids, logits):
        next_token_ids = torch.argmax(logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(-1)], dim=-1)
        return input_ids


class NumpyTensorAdapter():
    def randn(self, shape):
        return np.random.randn(*shape)

    def randint(self, low, high, size):
        return np.random.randint(low, high, size)

    def add_token_inputs_ids(self, input_ids, logits):
        next_token_ids = np.argmax(logits, axis=-1)
        print("next_token_ids",next_token_ids)
        input_ids = np.concatenate([input_ids, next_token_ids[..., None]], axis=-1)
        return input_ids


class MLXTensorAdapter():
    def __init__(self):
        import mlx
        self.mlx = mlx

    def randn(self, shape):
        return self.mlx.random.randn(*shape)

    def randint(self, low, high, size):
        return self.mlx.random.randint(low, high, size)

    def add_token_inputs_ids(self, input_ids, logits):
        next_token_ids = self.mlx.argmax(logits, axis=-1)
        input_ids = self.mlx.concatenate([input_ids, next_token_ids[..., None]], axis=-1)
        return input_ids

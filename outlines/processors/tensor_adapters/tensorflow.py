"""Tensor adapter for the `tensorflow` library."""

from outlines.processors.tensor_adapters.base import TensorAdapter


class TensorFlowTensorAdapter(TensorAdapter):
    library_name = "tensorflow"

    def __init__(self):
        import tensorflow as tf

        self.tf = tf

    def shape(self, tensor):
        return tensor.shape

    def unsqueeze(self, tensor):
        return self.tf.expand_dims(tensor, axis=0)

    def squeeze(self, tensor):
        return self.tf.squeeze(tensor, axis=0)

    def to_list(self, tensor):
        return tensor.numpy().tolist()

    def to_scalar(self, tensor):
        return tensor.numpy().item()

    def full_like(self, tensor, fill_value):
        return self.tf.fill(self.tf.shape(tensor), fill_value)

    def concatenate(self, tensors):
        return self.tf.concat(tensors, axis=0)

    def get_device(self, tensor):
        return tensor.device

    def to_device(self, tensor, device):
        # TensorFlow handles device placement differently
        with self.tf.device(device):
            return self.tf.identity(tensor)

    def boolean_ones_like(self, tensor):
        return self.tf.ones_like(tensor, dtype=self.tf.bool)

    def apply_mask(self, tensor, mask, value):
        return self.tf.where(mask, self.tf.constant(value, dtype=tensor.dtype), tensor)

    def argsort_descending(self, tensor):
        return self.tf.argsort(tensor, direction='DESCENDING')

    def create_end_thinking_bitmask(self, size, end_thinking_token_id):
        bitmask = self.tf.zeros(
            (size + 31) // 32,
            dtype=self.tf.int32
        )
        byte_index = end_thinking_token_id // 32
        bit_index = end_thinking_token_id % 32
        bitmask = self.tf.tensor_scatter_nd_update(
            bitmask,
            [[byte_index]],
            [1 << bit_index]
        )
        return bitmask

import mlx.core as mx
import numpy as np
import torch

from outlines.processors import OutlinesLogitsProcessor


def is_mlx_lm_allowed():
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return mx.metal.is_available()


class HalvingLogitsProcessor(OutlinesLogitsProcessor):
    """Simply halve the passed logits"""

    def process_logits(self, input_ids, logits):
        return logits / 2


class LogitsProcessorBenchmark:
    params = ["torch", "numpy"]
    if mx.metal.is_available():
        params += ["mlx"]

    def setup(self, array_library):
        self.logits_processor = HalvingLogitsProcessor()

        # logits: (4, 30,000 ) dtype=float
        # input_ids shape: (4, 2048) dtype=int
        if array_library == "torch":
            self.logits = torch.rand((4, 30000), dtype=torch.float)
            self.input_ids = torch.randint(
                low=0, high=30000, size=(4, 2048), dtype=torch.int
            )
        elif array_library == "numpy":
            self.logits = np.random.rand(4, 30000).astype(np.float32)
            self.input_ids = np.random.randint(low=0, high=30000, size=(4, 2048))
        elif array_library == "mlx":
            self.logits = mx.random.uniform(
                low=-1e9, high=1e9, shape=(4, 30000), dtype=mx.float32
            )
            self.input_ids = mx.random.randint(
                low=0, high=30000, shape=(4, 2048), dtype=mx.int32
            )
        else:
            raise ValueError

    def time_logits_processor(self, array_library):
        self.logits_processor(self.input_ids, self.logits)

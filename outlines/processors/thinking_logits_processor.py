from outlines.processors.base_logits_processor import OutlinesLogitsProcessor, TensorType


class ThinkingLogitsProcessor(OutlinesLogitsProcessor):

    def __init__(self, end_thinking_token_id: int, thinking_max_tokens: int, logits_processor: OutlinesLogitsProcessor):
        super().__init__(logits_processor.tensor_library_name)
        self.logits_processor = logits_processor
        self.end_thinking_token_id = end_thinking_token_id
        self.thinking_max_tokens = thinking_max_tokens
        self.is_first_token = True

    def reset(self) -> None:
        self.is_first_token = True
        self.logits_processor.reset()

    def setup(self, batch_size: int) -> None:
        self._is_thinking = [self.end_thinking_token_id is not None] * batch_size
        self._num_tokens_generated = 0

    def process_logits(self, input_ids: TensorType, logits: TensorType) -> TensorType:

        batch_size = self.tensor_adapter.shape(input_ids)[0]

        if self.is_first_token:
            self.setup(batch_size)
            self.is_first_token = False
        else:
            self._num_tokens_generated += 1
            for i in range(batch_size):
                if not self._is_thinking[i]:
                    continue
                latest_token_id = self.tensor_adapter.to_scalar(input_ids[i][-1])
                if latest_token_id == self.end_thinking_token_id:
                    self._is_thinking[i] = False
                elif self._num_tokens_generated >= self.thinking_max_tokens:
                    logits[i][self.end_thinking_token_id] = float("inf")

        if all(self._is_thinking):
            return logits

        return self.logits_processor.process_logits(input_ids, logits)

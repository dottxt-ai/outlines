from outlines.processors import fsm_logits_processor


class VLLMTokenizerAdapter:
    @staticmethod
    def _vllm_adapt_tokenizer(llm):
        """Adapt vLLM's tokenizer to use to compile the FSM.

        The API of Outlines tokenizers is slightly different to that of
        `transformers`. In addition we need to handle the missing spaces to
        Llama's tokenizer to be able to compile FSMs for this model.
        """
        tokenizer = llm.tokenizer.tokenizer
        tokenizer.vocabulary = tokenizer.get_vocab()
        tokenizer.special_tokens = set(tokenizer.all_special_tokens)

        def convert_token_to_string(token: str) -> str:
            from transformers.file_utils import SPIECE_UNDERLINE

            string = tokenizer.convert_tokens_to_string([token])

            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

            return string

        tokenizer.convert_token_to_string = convert_token_to_string

        return tokenizer


class RegexLogitsProcessor(fsm_logits_processor.RegexLogitsProcessor):
    _adapt_tokenizer = VLLMTokenizerAdapter._vllm_adapt_tokenizer


class JSONLogitsProcessor(fsm_logits_processor.JSONLogitsProcessor):
    _adapt_tokenizer = VLLMTokenizerAdapter._vllm_adapt_tokenizer


class CFGLogitsProcessor(fsm_logits_processor.JSONLogitsProcessor):
    _adapt_tokenizer = VLLMTokenizerAdapter._vllm_adapt_tokenizer

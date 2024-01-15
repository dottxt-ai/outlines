import re

import torch
from transformers import AutoTokenizer

from outlines.serve.vllm import RegexLogitsProcessor, _patched_apply_logits_processors


class MockModel:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")


def sample_from_logits(logits):
    probs = torch.exp(logits) / torch.sum(torch.exp(logits))
    return torch.multinomial(probs, 1).item()


def test_time_regexp():
    pattern = r"(0?[1-9]|1[0-2]):[0-5]\d\s?(am|pm)?"
    llm = MockModel()
    logits_processor = RegexLogitsProcessor(pattern, llm)

    token_ids = []
    while True:
        random_scores = -10 + 20 * torch.rand(len(llm.tokenizer.vocab))
        logits = logits_processor(
            seq_id=0,
            input_ids=token_ids,
            scores=random_scores,
        )
        new_token_id = sample_from_logits(logits)
        if new_token_id == llm.tokenizer.eos_token_id:
            break
        token_ids.append(new_token_id)

    assert re.fullmatch(pattern, llm.tokenizer.decode(token_ids)) is not None


def test_time_regexp_multiple_samples():
    num_seq = 64

    pattern = r"(0?[1-9]|1[0-2]):[0-5]\d\ ?(am|pm)?"
    llm = MockModel()

    class MockSeqData:
        def __init__(self):
            self.output_token_ids = []

    class MockSamplingParams:
        logits_processors = [RegexLogitsProcessor(pattern, llm)]

    class MockSamplingMeta:
        seq_groups = [[range(num_seq), MockSamplingParams()]]  # seq_ids
        seq_data = {seq_id: MockSeqData() for seq_id in range(num_seq)}

    sampling_meta = MockSamplingMeta()

    results = []
    while True:
        complete_seq_ids = set()

        logits = torch.randn(len(sampling_meta.seq_data), len(llm.tokenizer.vocab))
        new_logits = _patched_apply_logits_processors(logits, sampling_meta)
        seq_ids = sorted(sampling_meta.seq_groups[0][0])
        for logits_row, seq_id in zip(new_logits, seq_ids):
            new_token_id = sample_from_logits(logits_row)
            if new_token_id == llm.tokenizer.eos_token_id:
                complete_seq_ids.add(seq_id)
                results.append(sampling_meta.seq_data[seq_id].output_token_ids)
            else:
                sampling_meta.seq_data[seq_id].output_token_ids.append(new_token_id)

        if complete_seq_ids:
            seq_datas = [
                sd
                for seq_id, sd in sampling_meta.seq_data.items()
                if seq_id not in complete_seq_ids
            ]
            sampling_meta.seq_data = {
                i: seq_data for i, seq_data in enumerate(seq_datas)
            }
            sampling_meta.seq_groups[0][0] = range(len(sampling_meta.seq_data))

        if not sampling_meta.seq_data:
            break

    assert len(results) == num_seq
    for result in results:
        print(llm.tokenizer.decode(result))
        assert re.fullmatch(pattern, llm.tokenizer.decode(result)) is not None

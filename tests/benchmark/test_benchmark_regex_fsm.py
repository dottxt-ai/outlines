import pytest

import outlines

outlines.disable_cache()

from outlines.fsm.fsm import RegexFSM  # noqa: E402

regex_samples = {
    "email": r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
    "complex_phone": "\\+?\\d{1,4}?[-.\\s]?\\(?\\d{1,3}?\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}",
    "simple_phone": "\\+?[1-9][0-9]{7,14}",
    "date": r"([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])|([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])",
    "time": r"(0?[1-9]|1[0-2]):[0-5]\d\s?(am|pm)?",
    "ip": r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
    "url": r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "complex_span_constrained_relation_extraction": "(['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?\\s\\|\\s([^|\\(\\)\n]{1,})\\s\\|\\s['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?(\\s\\|\\s\\(([^|\\(\\)\n]{1,})\\s\\|\\s([^|\\(\\)\n]{1,})\\))*\\n)*",
}


@pytest.mark.parametrize("regex_name", regex_samples.keys())
def test_benchmark_regex_to_fsm(
    benchmark, tokenizer, ensure_numba_compiled, regex_name
):
    """Benchmark converting regex to FSM"""
    regex_str = regex_samples[regex_name]
    benchmark.pedantic(
        RegexFSM,
        args=(regex_str, tokenizer),
        rounds=8,
    )

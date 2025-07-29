"""Base class for all backends."""

from abc import ABC, abstractmethod
from typing import Any


LogitsProcessorType = Any


class BaseBackend(ABC):
    """Base class for all backends.

    The subclasses must implement methods that create a logits processor
    from a JSON schema, regex or CFG.

    """

    @abstractmethod
    def get_json_schema_logits_processor(
        self, json_schema: str
    ) -> LogitsProcessorType:
        """Create a logits processor from a JSON schema.

        Parameters
        ----------
        json_schema: str
            The JSON schema to create a logits processor from.

        Returns
        -------
        LogitsProcessorType
            The logits processor.

        """
        ...

    @abstractmethod
    def get_regex_logits_processor(self, regex: str) -> LogitsProcessorType:
        """Create a logits processor from a regex.

        Parameters
        ----------
        regex: str
            The regex to create a logits processor from.

        Returns
        -------
        LogitsProcessorType
            The logits processor.

        """
        ...

    @abstractmethod
    def get_cfg_logits_processor(self, grammar: str) -> LogitsProcessorType:
        """Create a logits processor from a context-free grammar.

        Parameters
        ----------
        grammar: str
            The context-free grammar to create a logits processor from.

        Returns
        -------
        LogitsProcessorType
            The logits processor.

        """
        ...

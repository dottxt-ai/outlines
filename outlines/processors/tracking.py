"""
A logit processor that enables analysis and debugging of logit biasing by tracking logits before and after processing.

This module provides the LogitTrackingProcessor class which wraps any other logit processor and tracks
both the original and processed logits during text generation. This is particularly useful for:

- Debugging logit processors by analyzing how they modify token probabilities
- Visualizing the effects of logit biasing on token distributions
- Understanding how constraints affect the generation process
- Validating that processors are working as intended

The tracking processor maintains a history of logits at each position in the generated sequence,
allowing for both real-time analysis and post-generation investigation. It can be configured to:

- Track all positions or maintain a sliding window of recent positions
- Track original logits, processed logits, or both
- Provide statistics and token-level analysis at any tracked position
- Reconstruct the generated sequence from tracked logits

Examples
--------
Basic regex tracking:
>>> from outlines.processors import RegexLogitsProcessor, LogitTrackingProcessor
>>> 
>>> # Create a regex processor and wrap it with tracking
>>> base_processor = RegexLogitsProcessor(r"[0-9]{4}", tokenizer)
>>> tracking_processor = LogitTrackingProcessor(base_processor)
>>> 
>>> # After generation, analyze the results
>>> tokens = tracking_processor.get_top_tokens(0)  # Get top token candidates at first position with original logits
>>> stats = tracking_processor.get_statistics(0)  # Get stats for first token
>>> seq = tracking_processor.get_sequence(3)  # Reconstruct sequence up to position 3

Analyzing structured output generation:
>>> from pydantic import BaseModel
>>> from outlines.processors import JSONLogitsProcessor
>>> 
>>> # Define a schema and create a tracking processor
>>> class WeatherResponse(BaseModel):
...     temperature: float
...     conditions: str
>>> 
>>> base_processor = JSONLogitsProcessor(WeatherResponse, tokenizer)
>>> tracking_processor = LogitTrackingProcessor(base_processor)
>>> 
>>> # After generation, analyze how the processor constrained the output
>>> stats = tracking_processor.get_statistics(0)
>>> print(f"Valid tokens after processing: {stats['processed']['valid_tokens']}")
>>> tokens = tracking_processor.get_top_tokens(0, k=5)
>>> print("Top 5 allowed tokens:", [t['token'] for t in tokens['processed']])

Notes
-----
- The processor stores logits in memory, so consider using max_positions for long sequences
- Original logits are stored before any processing is applied
- Processed logits will contain -inf values when structured outputs are used
- Token decoding requires the wrapped processor to have a tokenizer attribute
- Memory usage grows linearly with sequence length when max_positions is not set
- The processor can be used with any OutlinesLogitsProcessor implementation
"""
from typing import Dict, List, Optional, Union
import warnings
import torch

from .base_logits_processor import OutlinesLogitsProcessor, Array


class LogitTrackingProcessor(OutlinesLogitsProcessor):
    """A logit processor that wraps any other logit processor and tracks logits before and after processing.

    This processor saves both the original logits (before any bias is applied) and the processed logits
    (after bias is applied) for analysis. The logits are stored in dictionaries keyed by token position,
    where position 0 is the first generated token, position 1 is the second generated token, and so on.
    When max_positions is set, it maintains a sliding window of the most recent token positions.

    Attributes
    ----------
    processor : OutlinesLogitsProcessor
        The underlying logit processor to wrap
    original_logits : Dict[int, torch.Tensor]
        Dictionary mapping token position to original logits before processing.
        When max_positions is set, only the most recent positions are kept.
    processed_logits : Dict[int, torch.Tensor]
        Dictionary mapping token position to logits after processing.
        When max_positions is set, only the most recent positions are kept.
    max_positions : Optional[int]
        Maximum number of recent token positions to track. If None, tracks all positions.
        When set, maintains a sliding window of the most recent positions, removing
        older positions when the limit is reached.
    track_original : bool
        Whether to track original logits before processing
    track_processed : bool
        Whether to track processed logits after processing

    Examples
    --------
    Basic usage:
    >>> # Track all positions
    >>> processor = LogitTrackingProcessor(base_processor)
    >>> # Track only the 5 most recent positions
    >>> processor = LogitTrackingProcessor(base_processor, max_positions=5)
    >>> # After generating "2023":
    >>> # - Position 0 contains logits that generated "2"
    >>> # - Position 1 contains logits that generated "0"
    >>> # - Position 2 contains logits that generated "2"
    >>> # - Position 3 contains logits that generated "3"

    Accessing tracked logits:
    >>> # Get original logits for first token (position 0)
    >>> original_logits = processor.original_logits[0]  # shape: (vocab_size,)
    >>> # Get processed logits for second token (position 1)
    >>> processed_logits = processor.processed_logits[1]  # shape: (vocab_size,)
    >>> # Get all tracked positions
    >>> positions = sorted(processor.original_logits.keys())  # [0, 1, 2, 3]

    Analyzing logits:
    >>> # Get statistics for a position
    >>> stats = processor.get_statistics(0)
    >>> print(stats['original']['mean'])  # Mean of original logits
    >>> print(stats['processed']['valid_tokens'])  # Number of valid tokens after processing
    >>> # Get most likely tokens at a position
    >>> tokens = processor.get_top_tokens(0, k=5)  # Get top 5 tokens
    >>> print(tokens['original'])  # [{'token': '2', 'prob': 0.8}, ...]
    >>> print(tokens['processed'])  # [{'token': '2', 'prob': 1.0}, ...]
    >>> # Reconstruct sequence from tracked logits
    >>> seq = processor.get_sequence(3)  # Get sequence up to position 3
    >>> print(seq)  # "2023"
    >>> seq_orig = processor.get_sequence(3, use_processed=False)  # Using original logits
    >>> # Clear tracking data
    >>> processor.clear_tracking()
    """

    def __init__(
        self, 
        processor: OutlinesLogitsProcessor,
        max_positions: Optional[int] = None,
        track_original: bool = True,
        track_processed: bool = True,
    ):
        """Initialize the tracking processor.

        Parameters
        ----------
        processor : OutlinesLogitsProcessor
            The logit processor to wrap and track
        max_positions : Optional[int]
            Maximum number of recent positions to track. If None, tracks all positions.
            For example, if max_positions=5, only the 5 most recent positions will be
            kept in memory, with older positions being removed as new ones are added.
        track_original : bool
            Whether to track original logits
        track_processed : bool
            Whether to track processed logits
        
        Raises
        ------
        TypeError
            If processor is not an instance of OutlinesLogitsProcessor
        ValueError
            If max_positions is not None and <= 0
        """
        if not isinstance(processor, OutlinesLogitsProcessor):
            raise TypeError("processor must be an instance of OutlinesLogitsProcessor")
        
        if max_positions is not None and max_positions <= 0:
            raise ValueError("max_positions must be None or a positive integer")
        
        self.processor = processor
        self.max_positions = max_positions
        self.track_original = track_original
        self.track_processed = track_processed
        self.original_logits: Dict[int, torch.Tensor] = {}
        self.processed_logits: Dict[int, torch.Tensor] = {}

    def _update_tracking_dict(self, tracking_dict: Dict[int, torch.Tensor], pos: int, logits: torch.Tensor) -> None:
        """Update a tracking dictionary, maintaining the max_positions limit if set.
        
        Parameters
        ----------
        tracking_dict : Dict[int, torch.Tensor]
            The dictionary to update
        pos : int
            The position to add
        logits : torch.Tensor
            The logits to store
        """
        tracking_dict[pos] = logits.detach().clone()
        if self.max_positions is not None and len(tracking_dict) > self.max_positions:
            # Remove oldest position when limit is reached
            oldest_pos = min(tracking_dict.keys())
            del tracking_dict[oldest_pos]

    def process_logits(
        self, 
        input_ids: Array,
        logits: Array,
    ) -> Array:
        """Process logits and save both original and processed versions.

        Parameters
        ----------
        input_ids : Array
            The input token ids for each sequence in the batch
        logits : Array
            The original logits to process, shape (batch_size, vocab_size)

        Returns
        -------
        Array
            The processed logits, shape (batch_size, vocab_size)
        """
        # Ensure shapes match
        assert logits.shape[:-1] == torch.tensor(input_ids).shape[:-1], "batch dimensions must match"

        # Save original logits if tracking is enabled
        if self.track_original:
            for i, seq_ids in enumerate(input_ids):
                pos = len(seq_ids) - 1
                self._update_tracking_dict(self.original_logits, pos, logits[i])

        # Process logits using wrapped processor
        processed = self.processor.process_logits(input_ids, logits)

        # Save processed logits if tracking is enabled
        if self.track_processed:
            for i, seq_ids in enumerate(input_ids):
                pos = len(seq_ids) - 1
                self._update_tracking_dict(self.processed_logits, pos, processed[i])

        return processed

    def clear_tracking(self) -> None:
        """Clear all tracked logits to free memory."""
        self.original_logits.clear()
        self.processed_logits.clear()

    def get_statistics(self, pos: int) -> Dict[str, Dict[str, float]]:
        """Get basic statistics for the logits at a given position.

        Parameters
        ----------
        pos : int
            The position to get statistics for

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing statistics for both original and processed logits:
            {
                'original': {'mean': float, 'std': float, 'min': float, 'max': float},
                'processed': {'mean': float, 'std': float, 'min': float, 'max': float}
            }
        
        Raises
        ------
        KeyError
            If the position has not been tracked
        """
        stats = {}
        
        if pos in self.original_logits and self.track_original:
            orig = self.original_logits[pos]
            stats['original'] = {
                'mean': orig.mean().item(),
                'std': orig.std().item(),
                'min': orig.min().item(),
                'max': orig.max().item()
            }
            
        if pos in self.processed_logits and self.track_processed:
            proc = self.processed_logits[pos]
            valid_mask = proc != float('-inf')
            if valid_mask.any():
                valid_logits = proc[valid_mask]
                stats['processed'] = {
                    'mean': valid_logits.mean().item(),
                    'std': valid_logits.std().item(),
                    'min': valid_logits.min().item(),
                    'max': valid_logits.max().item(),
                    'valid_tokens': valid_mask.sum().item()
                }
            else:
                stats['processed'] = {
                    'valid_tokens': 0
                }
        
        if not stats:
            raise KeyError(f"Position {pos} has not been tracked")
            
        return stats

    def get_top_tokens(self, pos: int, k: int = 10) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """Get the top-k most likely tokens at a given position.

        Parameters
        ----------
        pos : int
            The position to get tokens for
        k : int, optional
            Number of top tokens to return, by default 10

        Returns
        -------
        Dict[str, List[Dict[str, Union[str, float]]]]
            Dictionary containing top tokens for both original and processed logits:
            {
                'original': [{'token': str, 'prob': float}, ...],
                'processed': [{'token': str, 'prob': float}, ...]
            }
            For processed logits, only tokens with non-inf logits are included.
        
        Examples
        --------
        >>> # Get top 5 tokens at position 0
        >>> tokens = processor.get_top_tokens(0, k=5)
        >>> # Original logits show multiple possibilities
        >>> print(tokens['original'])
        [{'token': '2', 'prob': 0.8}, {'token': '1', 'prob': 0.1}, ...]
        >>> # Processed logits may have fewer valid tokens due to constraints
        >>> print(tokens['processed'])
        [{'token': '2', 'prob': 1.0}]
        
        Raises
        ------
        KeyError
            If the position has not been tracked
        AttributeError
            If the processor's tokenizer is not accessible
        """
        if not hasattr(self.processor, 'tokenizer'):
            raise AttributeError("Cannot get tokens: processor has no tokenizer attribute")

        result = {}
        
        if pos in self.original_logits and self.track_original:
            orig = self.original_logits[pos]
            probs = torch.softmax(orig, dim=-1)
            top_k = torch.topk(probs, min(k, len(probs)))
            
            result['original'] = [
                {
                    'token': self.processor.tokenizer.decode([token_id.item()]),
                    'prob': prob.item()
                }
                for token_id, prob in zip(top_k.indices, top_k.values)
            ]
            
        if pos in self.processed_logits and self.track_processed:
            proc = self.processed_logits[pos]
            valid_mask = proc != float('-inf')
            if valid_mask.any():
                # Create a copy and zero out -inf values for softmax
                proc_valid = proc.clone()
                proc_valid[~valid_mask] = -1e9  # Very negative but not -inf
                probs = torch.softmax(proc_valid, dim=-1)
                # Zero out invalid token probabilities
                probs[~valid_mask] = 0
                
                top_k = torch.topk(probs, min(k, valid_mask.sum().item()))
                
                result['processed'] = [
                    {
                        'token': self.processor.tokenizer.decode([token_id.item()]),
                        'prob': prob.item()
                    }
                    for token_id, prob in zip(top_k.indices, top_k.values)
                    if prob > 0  # Only include tokens with non-zero probability
                ]
            else:
                result['processed'] = []  # No valid tokens
        
        if not result:
            raise KeyError(f"Position {pos} has not been tracked")
            
        return result

    def get_sequence(self, end_pos: int, use_processed: bool = True) -> str:
        """Reconstruct the sequence up to a given position by taking the most likely token at each position.

        Parameters
        ----------
        end_pos : int
            The position to reconstruct up to (inclusive)
        use_processed : bool, optional
            Whether to use processed logits (True) or original logits (False), by default True

        Returns
        -------
        str
            The reconstructed sequence

        Examples
        --------
        >>> # Get sequence up to position 3 using processed logits
        >>> seq = processor.get_sequence(3)
        >>> print(seq)  # "2023"
        >>> # Compare with sequence from original logits
        >>> seq_orig = processor.get_sequence(3, use_processed=False)
        >>> print(seq_orig)  # Might be different if processor constrained choices
        >>> # Get partial sequence
        >>> seq_partial = processor.get_sequence(1)
        >>> print(seq_partial)  # "20"

        Raises
        ------
        KeyError
            If any position up to end_pos has not been tracked
        AttributeError
            If the processor's tokenizer is not accessible
        ValueError
            If end_pos is negative
        """
        if not hasattr(self.processor, 'tokenizer'):
            raise AttributeError("Cannot get sequence: processor has no tokenizer attribute")
        
        if end_pos < 0:
            raise ValueError("end_pos must be non-negative")

        # Check if we have all positions up to end_pos
        logits_dict = self.processed_logits if use_processed else self.original_logits
        for pos in range(end_pos + 1):
            if pos not in logits_dict:
                raise KeyError(f"Position {pos} has not been tracked")

        # Get the most likely token at each position
        tokens = []
        for pos in range(end_pos + 1):
            logits = logits_dict[pos]
            if use_processed:
                # Handle -inf values for processed logits
                valid_mask = logits != float('-inf')
                if not valid_mask.any():
                    continue  # Skip positions with no valid tokens
                logits = logits.masked_fill(~valid_mask, -1e9)

            # Get the most likely token
            token_id = torch.argmax(logits).item()
            token = self.processor.tokenizer.decode([token_id])
            tokens.append(token)

        return ''.join(tokens) 


def add_tracking(generator, max_positions: Optional[int] = None) -> "Generator":
    """Add logit tracking to an existing generator.
    
    This is a convenience function that wraps a generator's logits processor with
    a LogitTrackingProcessor, enabling analysis of token probabilities during generation.

    Parameters
    ----------
    generator : Generator
        The generator to add tracking to
    max_positions : Optional[int]
        Maximum number of positions to track. If None, tracks all positions.
        For long sequences, consider setting this to limit memory usage.
    
    Returns
    -------
    Generator
        The same generator with tracking enabled

    Examples
    --------
    >>> from outlines.generate import json
    >>> from outlines.processors import add_tracking
    >>> 
    >>> generator = json(model, schema)
    >>> generator = add_tracking(generator)  # Enable tracking
    >>> 
    >>> # For long sequences, limit tracking to recent positions
    >>> generator = add_tracking(generator, max_positions=5)
    """
    generator.logits_processor = LogitTrackingProcessor(
        generator.logits_processor,
        max_positions=max_positions
    )
    return generator 
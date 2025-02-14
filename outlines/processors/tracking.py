"""
A simple logit processor that tracks probabilities for both structured and unstructured generation.

For each token generated, we store:
- The raw logits the model would assign naturally
- The filtered logits after applying structural constraints
- A mapping from vocabulary indices to token strings
"""
from typing import TYPE_CHECKING, Optional, Union, List, Literal, Dict, Any

import numpy as np
import torch
from numpy.typing import NDArray

from .base_logits_processor import OutlinesLogitsProcessor, Array

if TYPE_CHECKING:
    from outlines.generate import Generator

# Try importing pandas, but don't fail if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = Any  # For type hints when pandas is not available

class LogitTrackingProcessor(OutlinesLogitsProcessor):
    """Tracks logits for both structured and unstructured token generation.

    For each position in the sequence, stores:
    - unstructured_logits: Raw logits from the model
    - structured_logits: Logits after applying constraints
    - vocab_tokens: Mapping from vocab indices to token strings
    - chosen_tokens: Track actual sampled token IDs during generation

    Each logit matrix has:
    - Columns: One for each position in the generated sequence
    - Rows: One for each token in the vocabulary
    
    Attributes
    ----------
    processor : Optional[OutlinesLogitsProcessor]
        The processor that applies structural constraints
    unstructured_logits : List[NDArray]
        Raw logits from the model for each position
    structured_logits : List[NDArray]
        Logits after applying constraints for each position
    vocab_tokens : Optional[List[str]]
        Mapping from vocabulary indices to token strings
    chosen_tokens : List[int]
        Track actual chosen token IDs during generation. This is used 
        to ensure a log of the tokens the model generates, and is
        used internally for various convenience functions.
    _started : bool
        Tracks whether to start appending chosen tokens. This is set to True
        on the first call to process_logits, and remains True thereafter.
        
    """
    
    def __init__(self, processor=None):
        """Initialize the tracking processor.
        
        Parameters
        ----------
        processor : Optional[OutlinesLogitsProcessor]
            The processor that applies structural constraints.
            If None, only tracks raw logits.
        """
        self.processor = processor
        self.unstructured_logits = []  # List of logit arrays, one per position
        self.structured_logits = []    # List of logit arrays, one per position
        self.vocab_tokens = None      # Will store the vocabulary mapping
        self.chosen_tokens = []       # Track actual chosen tokens during generation
        self._started = False         # Tracks whether to start appending chosen tokens
        
    def process_logits(self, input_ids: Array, logits: Array) -> Array:
        """Process logits and store them.
        
        This method:
        1. Stores the raw logits from the model
        2. Applies any structural constraints if a processor exists
        3. Stores the constrained logits
        4. Tracks the chosen token ID
        
        Parameters
        ----------
        input_ids : Array
            The input token ids for the sequence. Must be single batch.
        logits : Array
            The original logits to process, shape (1, vocab_size)
            
        Returns
        -------
        Array
            The processed logits, shape (1, vocab_size)
            
        Raises
        ------
        ValueError
            If batch size > 1 is provided. The tracking processor currently
            only supports single-batch processing.
        """
        # Enforce single batch processing
        if logits.shape[0] > 1:
            raise ValueError(
                "LogitTrackingProcessor only supports single-batch processing. "
                f"Got batch size {logits.shape[0]}"
            )
        if len(input_ids) > 1:
            raise ValueError(
                "LogitTrackingProcessor only supports single-batch processing. "
                f"Got {len(input_ids)} sequences"
            )

        # Always store the raw logits as unstructured
        self.unstructured_logits.append(logits[0].detach().cpu().numpy().copy())
        
        # Store the actual chosen token ID if available
        if self._started and len(input_ids[0]) > 1:
            # Get the last token from the current sequence
            self.chosen_tokens.append(input_ids[0][-1])

        # If we haven't started tracking yet, do so now.
        # this will only happen on the first call to process_logits.
        else:
            self._started = True

        # Apply structural constraints if we have a processor
        if self.processor is not None:
            processed = self.processor.process_logits(input_ids, logits)
            self.structured_logits.append(processed[0].detach().cpu().numpy().copy())
            return processed
            
        # For unconstrained generation, structured = unstructured
        self.structured_logits.append(logits[0].detach().cpu().numpy().copy())
        return logits
            
    def get_probabilities(self, as_matrix: bool = False) -> Dict[str, Union[List[NDArray], NDArray]]:
        """Get probability distributions computed from stored logits.
        
        Parameters
        ----------
        as_matrix : bool
            If True, convert probability lists to matrices.
            Each matrix will have shape (vocab_size, n_positions), i.e. 
            return_value['unstructured'] is a vocab_size x n_positions matrix.

            If False, return a list of n_positions arrays, each array having
            shape (vocab_size,).
        
        Returns
        -------
        Dict[str, Union[List[NDArray], NDArray]]
            Contains a dictionary with two keys:
            - unstructured: Raw probability distributions
            - structured: Probability distributions after constraints
            Each can be either a list of arrays or a single matrix
        """
        # Convert logits to probabilities
        unstructured_probs = [
            torch.softmax(torch.tensor(logits), dim=-1).numpy()
            for logits in self.unstructured_logits
        ]
        structured_probs = [
            torch.softmax(torch.tensor(logits), dim=-1).numpy()
            for logits in self.structured_logits
        ]
        
        if as_matrix:
            # Stack arrays into matrices
            unstructured = np.column_stack(unstructured_probs)
            structured = np.column_stack(structured_probs)
        else:
            # Return as lists
            unstructured = unstructured_probs
            structured = structured_probs
            
        return {
            'unstructured': unstructured,
            'structured': structured
        }

    def get_logits(self, as_matrix: bool = False) -> Dict[str, Union[List[NDArray], NDArray]]:
        """Get the stored logit values.
        
        Parameters
        ----------
        as_matrix : bool
            If True, convert logit lists to matrices.
            Each matrix will have shape (vocab_size, n_positions), i.e.
            return_value['unstructured'] is a vocab_size x n_positions matrix.

            If False, return a list of n_positions arrays, each array having
            shape (vocab_size,).
        
        Returns
        -------
        Dict[str, Union[List[NDArray], NDArray]]
            Contains a dictionary with two keys:
            - unstructured: Raw logit values
            - structured: Logit values after constraints
            Each can be either a list of arrays or a single matrix
        """
        if as_matrix:
            unstructured = np.column_stack(self.unstructured_logits)
            structured = np.column_stack(self.structured_logits)
        else:
            unstructured = self.unstructured_logits
            structured = self.structured_logits
            
        return {
            'unstructured': unstructured,
            'structured': structured
        }
        
    def get_top_tokens(
        self,
        k: int = 10,
        positions: Optional[Union[int, List[int]]] = None,
        include_logits: bool = True
    ) -> List[Dict[str, Any]]:
        """Get the top k tokens at specified positions with their probabilities and logits.
        
        Parameters
        ----------
        k : int, optional
            Number of top tokens to return, by default 10
        positions : Union[int, List[int]], optional
            Position(s) to analyze. Can be a single position or list of positions.
            By default analyzes all positions.
        include_logits : bool, optional
            Whether to include raw logit values in addition to probabilities
            
        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries, one per position, containing:
            - position: Position in sequence
            - text_so_far: Text generated up to this position
            - tokens: List of top k token dictionaries, each containing:
                - token: The token string
                - natural_prob: Unconstrained probability
                - constrained_prob: Probability after constraints
                - natural_logit: Raw logit value (if include_logits=True)
                - constrained_logit: Constrained logit value (if include_logits=True)
                - is_chosen: Whether this token was actually chosen
        """
        # Convert single position to list
        if positions is None:
            positions = list(range(len(self.structured_logits)))
        elif isinstance(positions, int):
            positions = [positions]
            
        # Get probabilities and logits
        probs = self.get_probabilities()
        logits = self.get_logits() if include_logits else None
        
        # Get vocab mapping
        vocab = self.get_vocab_mapping()
        
        results = []
        for pos in positions:
            if pos >= len(self.unstructured_logits):
                continue
                
            # Get text generated so far
            text_so_far = self.sequence(pos)
            
            # Get values for this position
            u_probs = probs['unstructured'][pos]
            s_probs = probs['structured'][pos]
            
            if include_logits:
                u_logits = logits['unstructured'][pos]
                s_logits = logits['structured'][pos]
            
            # Get top k indices by maximum probability
            top_indices = np.argsort(np.maximum(u_probs, s_probs))[-k:][::-1]
            
            # Get the actual next token for comparison
            next_token = self.sequence(pos + 1)[len(text_so_far):] if pos < len(self.structured_logits)-1 else ""
            
            # Build token info list
            tokens = []
            for idx in top_indices:
                token = vocab[idx]
                token_info = {
                    'token': token,
                    'unstructured_prob': float(u_probs[idx]),
                    'structured_prob': float(s_probs[idx]),
                    'is_chosen': token == next_token
                }
                
                if include_logits:
                    token_info.update({
                        'unstructured_logit': float(u_logits[idx]),
                        'structured_logit': float(s_logits[idx])
                    })
                    
                tokens.append(token_info)
            
            results.append({
                'position': pos,
                'text_so_far': text_so_far,
                'tokens': tokens
            })
            
        return results

    def get_vocab_mapping(self) -> List[str]:
        """Get the mapping from vocabulary indices to token strings. Each token
        matches 
        
        Returns
        -------
        List[str]
            List of token strings, where index matches vocabulary index
        
        Raises
        ------
        AttributeError
            If no tokenizer is available
        """
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("No tokenizer available for mapping tokens")
            
        if self.vocab_tokens is None:
            # Create the mapping if we haven't yet
            self.vocab_tokens = [
                self.processor.tokenizer.decode([i])[0]
                for i in range(len(self.unstructured_logits[0]))
            ]
            
        return self.vocab_tokens
        
    def clear(self):
        """Clear all stored logits."""
        self.unstructured_logits = []
        self.structured_logits = []
        self.chosen_tokens = []

    def to_dataframe(
        self,
        show: Literal["probs", "logits"] = "probs",
        min_value: Optional[float] = None
    ) -> "pd.DataFrame":
        """Convert tracking data to a pandas DataFrame for analysis.
        
        Parameters
        ----------
        show : Literal["probs", "logits"], optional
            Whether to show probabilities or logit values, by default "probs"
        min_value : Optional[float], optional
            If provided, only include tokens with values >= min_value
            in either structured or unstructured distribution
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - position: Token position in sequence
            - token: String representation of token
            - natural: Raw model values (probs/logits)
            - constrained: Values after constraints
            - chosen: Whether this token was chosen (True/False)

        Examples
        --------
        >>> # Get probability data for top 10 tokens
        >>> df = processor.to_dataframe(show="probs")
        >>> df.sort_values("natural", ascending=False).head()
        >>>
        >>> # Get logit data above threshold
        >>> df = processor.to_dataframe(show="logits", min_value=-5)
        >>> df.query("position == 0").nlargest(5, "natural")
        >>>
        >>> # Get all tokens with probability > 1%
        >>> df = processor.to_dataframe(show="probs", min_value=0.01)
            
        Raises
        ------
        ImportError
            If pandas is not installed
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for DataFrame support. "
                "Please install it with: pip install pandas"
            )
            
        # Get values based on show parameter
        if show == "probs":
            values = self.get_probabilities()
        else:
            values = self.get_logits()
            
        # Get vocab mapping
        vocab = self.get_vocab_mapping()
        
        # Create lists to store data
        rows = []
        
        # Process each position
        for pos in range(len(self.unstructured_logits)):
            u_vals = values['unstructured'][pos]
            s_vals = values['structured'][pos]
            
            # Get the chosen token at this position if available
            chosen_token = vocab[self.chosen_tokens[pos]] if pos < len(self.chosen_tokens) else None
            
            # Get indices to include based on filters
            if min_value is not None:
                # Get maximum value between structured/unstructured for sorting
                max_vals = np.maximum(u_vals, s_vals)
                
                # Both filters: get top k among values >= min_value
                valid_indices = np.where(max_vals >= min_value)[0]
                valid_indices = valid_indices[np.argsort(max_vals[valid_indices])[-10:]]
            else:
                # No filters: include all tokens
                valid_indices = range(len(vocab))
            
            # Add rows for valid indices
            for idx in valid_indices:
                token = vocab[idx]
                rows.append({
                    'position': pos,
                    'token': token,
                    'natural': u_vals[idx],
                    'constrained': s_vals[idx],
                    'chosen': token == chosen_token
                })
        
        return pd.DataFrame(rows)

    def sequence(self, pos: Optional[int] = None) -> str:
        """Get the sequence of tokens generated up to a position.
        
        Parameters
        ----------
        pos : Optional[int], optional
            Position to reconstruct up to (exclusive).
            If None, returns the entire sequence.
            
        Returns
        -------
        str
            The concatenated string of chosen tokens
            
        Raises
        ------
        AttributeError
            If no tokenizer is available for decoding
        """
        if not self.chosen_tokens:
            return ""
            
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("No tokenizer available for decoding sequence")
            
        # Get the tokenizer
        if hasattr(self.processor, 'tokenizer'):
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.tokenizer
            
        # Get tokens up to the specified position
        end_pos = len(self.chosen_tokens) if pos is None else pos
        tokens_to_decode = self.chosen_tokens[:end_pos]
        
        # Decode the sequence
        return "".join(tokenizer.decode(tokens_to_decode))


def add_tracking(generator: "Generator") -> "Generator":
    """Add probability tracking to any generator.
    
    This is a convenience function that wraps a generator's logits processor
    with a LogitTrackingProcessor, enabling analysis of token probabilities
    and logits during generation.

    Currently only works with structured generators, outlines.generate.text
    is not supported.
    
    Parameters
    ----------
    generator : Generator
        The generator to add tracking to
        
    Returns
    -------
    Generator
        The same generator with tracking enabled
        
    Examples
    --------
    >>> # Track probabilities for unconstrained text generation
    >>> generator = generate.text(model)
    >>> generator = add_tracking(generator)
    >>>
    >>> # Track probabilities for JSON generation
    >>> generator = generate.json(model, schema)
    >>> generator = add_tracking(generator)
    """
    # If there's no logits_processor, throw an error. Logit tracking
    # is currently only supported for structured generators.
    if generator.logits_processor is None:
        raise ValueError("Logit tracking is not supported for this generator")

    # Create tracking processor, wrapping any existing processor
    tracking = LogitTrackingProcessor(generator.logits_processor)

    # Add tokenizer for token mapping
    if hasattr(generator.logits_processor, 'tokenizer'):
        tracking.tokenizer = generator.logits_processor.tokenizer
    
    # Set as the generator's processor
    generator.logits_processor = tracking
    
    return generator

"""Tests for the LogitTrackingProcessor functionality.

This test suite verifies the behavior of the LogitTrackingProcessor, which wraps other
logit processors to track and analyze their behavior. The tests cover:

1. Basic functionality:
   - Initialization with different parameters
   - Logit processing and tracking
   - Statistics calculation
   - Token retrieval and sequence reconstruction

2. Error handling:
   - Invalid inputs
   - Missing tokenizer
   - Shape mismatches
   - Invalid positions

3. Memory management:
   - Sliding window tracking
   - Position limits
   - Clearing tracking data

4. Edge cases:
   - Empty sequences
   - All invalid logits
   - Large token requests
"""
import pytest
import torch
from typing import List

from outlines.processors.tracking import LogitTrackingProcessor, add_tracking
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor


class MockProcessor(OutlinesLogitsProcessor):
    """Mock processor that modifies logits in a predictable way."""
    def __init__(self):
        self.tokenizer = MockTokenizer()

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        # For testing purposes, set every other logit to -inf
        processed = logits.clone()
        processed[:, ::2] = float('-inf')
        return processed


class MockTokenizer:
    """Mock tokenizer for testing."""
    def decode(self, token_ids):
        return str(token_ids[0])


@pytest.fixture
def processor():
    """Fixture for creating a tracking processor with a mock base processor."""
    base = MockProcessor()
    return LogitTrackingProcessor(base)


def test_initialization():
    """Test initialization with various parameters."""
    base = MockProcessor()
    
    # Basic initialization
    processor = LogitTrackingProcessor(base)
    assert processor.processor == base
    assert processor.max_positions is None
    
    # With max positions
    processor = LogitTrackingProcessor(base, max_positions=5)
    assert processor.max_positions == 5
    
    # With tracking options
    processor = LogitTrackingProcessor(base, track_original=False)
    assert not processor.track_original
    assert processor.track_processed
    
    processor = LogitTrackingProcessor(base, track_processed=False)
    assert processor.track_original
    assert not processor.track_processed


@pytest.mark.parametrize("max_positions", [None, 1, 3, 5])
def test_position_tracking(processor, max_positions):
    """Test tracking with different max_positions values."""
    if max_positions is not None:
        processor.max_positions = max_positions
    
    # Process 5 positions
    for i in range(5):
        logits = torch.randn(1, 10)
        input_ids = [[j for j in range(i + 1)]]
        processor.process_logits(input_ids, logits)
    
    expected_positions = range(max(0, 5 - (max_positions or 5)), 5)
    assert all(pos in processor.original_logits for pos in expected_positions)
    assert all(pos in processor.processed_logits for pos in expected_positions)


@pytest.mark.parametrize("batch_size,vocab_size", [(1, 10), (2, 100)])
def test_logit_processing(processor, batch_size, vocab_size):
    """Test logit processing with different batch and vocab sizes."""
    input_ids = [[0] for _ in range(batch_size)]
    # Use deterministic logits instead of random ones
    logits = torch.ones(batch_size, vocab_size)
    
    processed = processor.process_logits(input_ids, logits)
    
    # Check tracking
    assert 0 in processor.original_logits
    assert 0 in processor.processed_logits
    
    # Check original logits preserved
    assert torch.allclose(processor.original_logits[0], logits[0])
    
    # Check processing (every other logit should be -inf)
    assert torch.all(torch.isinf(processed[:, ::2]))
    assert not torch.any(torch.isinf(processed[:, 1::2]))


def test_statistics(processor):
    """Test statistics calculation."""
    logits = torch.tensor([[1.0, 2.0, -1.0, 0.0]])
    processor.process_logits([[0]], logits)
    
    stats = processor.get_statistics(0)
    
    # Check original stats
    assert set(stats['original'].keys()) == {'mean', 'std', 'min', 'max'}
    assert abs(stats['original']['mean'] - 0.5) < 1e-6
    
    # Check processed stats
    assert 'valid_tokens' in stats['processed']
    assert stats['processed']['valid_tokens'] == 2  # Half should be valid


def test_top_tokens(processor):
    """Test top token retrieval."""
    logits = torch.tensor([[3.0, -1.0, 2.0, 0.0]])
    processor.process_logits([[0]], logits)
    
    tokens = processor.get_top_tokens(0, k=2)
    
    # Check structure
    assert set(tokens.keys()) == {'original', 'processed'}
    assert len(tokens['original']) == 2
    assert all(set(t.keys()) == {'token', 'prob'} for t in tokens['original'])
    
    # Check ordering
    assert tokens['original'][0]['prob'] > tokens['original'][1]['prob']


def test_sequence_reconstruction(processor):
    """Test sequence reconstruction from tracked logits."""
    # Generate a sequence
    for i in range(3):
        logits = torch.zeros(1, 10)
        logits[0, i] = 2.0  # Make token i most likely
        processor.process_logits([[j for j in range(i + 1)]], logits)
    
    # Test reconstruction
    seq = processor.get_sequence(2)
    assert len(seq) == 3
    
    # Test with original logits
    seq_orig = processor.get_sequence(2, use_processed=False)
    assert len(seq_orig) == 3


def test_clear_tracking(processor):
    """Test clearing tracked data."""
    processor.process_logits([[0]], torch.randn(1, 10))
    assert len(processor.original_logits) > 0
    
    processor.clear_tracking()
    assert len(processor.original_logits) == 0
    assert len(processor.processed_logits) == 0


def test_add_tracking_helper():
    """Test the add_tracking convenience function."""
    class MockGenerator:
        def __init__(self):
            self.logits_processor = MockProcessor()
    
    generator = MockGenerator()
    tracked = add_tracking(generator)
    
    assert isinstance(tracked.logits_processor, LogitTrackingProcessor)
    assert tracked.logits_processor.max_positions is None
    
    tracked = add_tracking(generator, max_positions=5)
    assert tracked.logits_processor.max_positions == 5


@pytest.mark.parametrize("invalid_value", [
    "not a processor",  # Invalid processor type
    0,  # Invalid max_positions
    -1,  # Negative max_positions
])
def test_invalid_inputs(invalid_value):
    """Test handling of invalid inputs."""
    base = MockProcessor()
    
    if isinstance(invalid_value, str):
        # Test invalid processor type
        processor = LogitTrackingProcessor(invalid_value)
        with pytest.raises(AttributeError, match="'str' object has no attribute 'process_logits'"):
            # Should raise AttributeError when trying to use process_logits
            processor.process_logits([[0]], torch.ones(1, 10))
    else:
        # Test invalid max_positions
        with pytest.raises(ValueError):
            LogitTrackingProcessor(base, max_positions=invalid_value)


def test_tracking_disabled():
    """Test behavior when tracking is disabled."""
    base = MockProcessor()
    processor = LogitTrackingProcessor(base, track_original=False, track_processed=False)
    
    # Process some logits
    logits = torch.ones(1, 10)
    processor.process_logits([[0]], logits)
    
    # Check that nothing was tracked
    assert len(processor.original_logits) == 0
    assert len(processor.processed_logits) == 0
    
    # Check that accessing stats raises KeyError
    with pytest.raises(KeyError):
        processor.get_statistics(0)


def test_missing_tokenizer():
    """Test behavior when processor has no tokenizer."""
    class ProcessorWithoutTokenizer(OutlinesLogitsProcessor):
        def process_logits(self, input_ids, logits):
            return logits
    
    processor = LogitTrackingProcessor(ProcessorWithoutTokenizer())
    processor.process_logits([[0]], torch.ones(1, 10))
    
    with pytest.raises(AttributeError):
        processor.get_top_tokens(0)


def test_shape_mismatch():
    """Test error handling for shape mismatch between input_ids and logits."""
    base = MockProcessor()
    processor = LogitTrackingProcessor(base)
    
    with pytest.raises(AssertionError, match="batch dimensions must match"):
        processor.process_logits([[0]], torch.ones(2, 10))  # batch size mismatch


def test_empty_sequence():
    """Test handling of empty input sequences."""
    base = MockProcessor()
    processor = LogitTrackingProcessor(base)
    
    logits = torch.ones(1, 10)
    processor.process_logits([[]], logits)
    
    # Position should be -1 for empty sequence
    assert -1 in processor.original_logits
    assert -1 in processor.processed_logits


def test_sliding_window_contents():
    """Test exact contents of sliding window tracking."""
    base = MockProcessor()
    processor = LogitTrackingProcessor(base, max_positions=2)
    
    # Process 4 positions
    for i in range(4):
        logits = torch.full((1, 10), float(i))  # Fill with position number for easy checking
        processor.process_logits([[j for j in range(i + 1)]], logits)
    
    # Should only have positions 2 and 3
    assert set(processor.original_logits.keys()) == {2, 3}
    assert torch.all(processor.original_logits[2][0] == 2.0)
    assert torch.all(processor.original_logits[3][0] == 3.0)


def test_sequence_reconstruction_all_invalid():
    """Test sequence reconstruction when all logits are invalid."""
    base = MockProcessor()
    processor = LogitTrackingProcessor(base)
    
    # Create logits where processed version has all -inf
    logits = torch.ones(1, 10)
    processor.process_logits([[0]], logits)
    processor.processed_logits[0].fill_(float('-inf'))
    
    # Should skip invalid positions
    seq = processor.get_sequence(0)
    assert seq == ""


def test_top_tokens_large_k():
    """Test get_top_tokens with k larger than vocab size."""
    base = MockProcessor()
    processor = LogitTrackingProcessor(base)
    
    vocab_size = 10
    logits = torch.ones(1, vocab_size)
    processor.process_logits([[0]], logits)
    
    # Request more tokens than exist
    tokens = processor.get_top_tokens(0, k=20)
    assert len(tokens['original']) == vocab_size
    assert len(tokens['processed']) <= vocab_size


@pytest.mark.parametrize("pos", [-1, -100])
def test_negative_position_access(pos):
    """Test accessing positions with negative indices."""
    base = MockProcessor()
    processor = LogitTrackingProcessor(base)
    
    with pytest.raises(ValueError, match="must be non-negative"):
        processor.get_sequence(pos)


@pytest.mark.parametrize("pos", [100, 1000])
def test_nonexistent_position_access(pos):
    """Test accessing statistics/tokens for non-existent positions."""
    base = MockProcessor()
    processor = LogitTrackingProcessor(base)
    
    with pytest.raises(KeyError):
        processor.get_statistics(pos)
    
    with pytest.raises(KeyError):
        processor.get_top_tokens(pos)
    
    with pytest.raises(KeyError):
        processor.get_sequence(pos) 
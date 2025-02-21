"""
Demo script showing how to use the LogitTrackingProcessor to analyze token probabilities.

This script demonstrates:
1. How language models naturally choose tokens
2. How structural constraints (like JSON or regex) affect these choices
3. Visualization of probability distributions
4. Analysis of token selection patterns
"""
from typing import Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import outlines.models as models
import outlines.generate as generate
from outlines.processors.tracking import track_logits, LogitTrackingProcessor
from outlines.processors import RegexLogitsProcessor
import transformers

def plot_token_distributions(tracking_processor, k=10, positions=None, prefix=""):
    """Plot token probability distributions before and after applying constraints.

    Creates a horizontal bar chart showing:
    - Blue bars: What tokens the model would naturally choose
    - Orange bars: What tokens are allowed by structural constraints

    Parameters
    ----------
    tracking_processor : LogitTrackingProcessor
        The processor containing tracked probabilities
    k : int, optional
        Number of top tokens to show in each plot, by default 10
    positions : List[int], optional
        Which positions to plot. If None, plots all positions.
    prefix : str, optional
        Prefix for the output filename

    Notes
    -----
    - Bar height indicates probability (how likely the model thinks each token is)
    - Tokens are sorted by maximum probability across both distributions
    - Only probabilities > 1% show their exact values
    - Grid lines help compare probabilities between tokens
    """
    # Get probability matrices and vocab mapping
    probs = tracking_processor.get_probabilities()
    vocab = tracking_processor.get_vocab_mapping()

    # Determine positions to plot
    if positions is None:
        positions = list(range(probs['unstructured'].shape[1]))
    n_positions = len(positions)

    # Create plot
    fig, axes = plt.subplots(1, n_positions, figsize=(7 * n_positions, 10))
    if n_positions == 1:
        axes = [axes]

    for idx, pos in enumerate(positions):
        # Get probabilities for this position
        unstructured = probs['unstructured'][:, pos]
        structured = probs['structured'][:, pos]

        # Get top k tokens by maximum probability
        top_indices = np.argsort(np.maximum(unstructured, structured))[-k:]

        # Create bar positions
        y = np.arange(len(top_indices))
        height = 0.35

        # Plot bars
        axes[idx].barh(y - height/2, unstructured[top_indices], height,
                      label='Natural Choice', alpha=0.7, color='skyblue')
        axes[idx].barh(y + height/2, structured[top_indices], height,
                      label='After Constraints', alpha=0.7, color='orange')

        # Customize plot
        axes[idx].set_title(f'Token {pos+1} in Sequence')
        axes[idx].set_yticks(y)
        axes[idx].set_yticklabels([vocab[i] for i in top_indices])
        axes[idx].set_xlabel('Probability')

        # Add legend
        axes[idx].legend(loc='upper right', bbox_to_anchor=(1, 1.1))
        axes[idx].grid(True, alpha=0.3)

        # Add probability values
        for i, (v1, v2) in enumerate(zip(unstructured[top_indices], structured[top_indices])):
            if v1 > 0.01:  # Only show probabilities > 1%
                axes[idx].text(v1 + 0.01, i - height/2, f'{v1:.1%}', va='center')
            if v2 > 0.01:
                axes[idx].text(v2 + 0.01, i + height/2, f'{v2:.1%}', va='center')

    plt.tight_layout()
    plt.savefig(f"{prefix}token_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap(tracking_processor, k=50, positions=None, prefix="", show_both=True, kind="logits", show_tokens=True):
    """Plot a heatmap of token probabilities across sequence positions.

    Creates a heatmap visualization showing how token probabilities evolve
    across different positions in the sequence. Optionally shows both
    natural and constrained probabilities side by side.

    Parameters
    ----------
    tracking_processor : LogitTrackingProcessor
        The processor containing tracked probabilities
    k : int, optional
        Number of top tokens to include in the heatmap, by default 50
    positions : List[int], optional
        Which positions to plot. If None, plots all positions.
    prefix : str, optional
        Prefix for the output filename
    show_both : bool, optional
        If True, shows both natural and constrained probabilities side by side.
        If False, only shows natural probabilities.
    kind : str, optional
        Whether to plot logits or probabilities, by default "logits"
    show_tokens : bool, optional
        Whether to show the token strings on the y-axis, by default True

    Notes
    -----
    - Brighter colors indicate higher probabilities
    - Y-axis shows token strings
    - X-axis shows position in sequence
    - Near-zero probabilities are masked out (shown in gray)
    - For constrained generation, blocked tokens appear masked
    """
    # Get probability matrices and vocab mapping
    if kind == "logits":
        things = tracking_processor.get_logits()
        # For logits, mask out very negative values
        threshold = -10  # Logits below this are effectively zero probability
    else:
        things = tracking_processor.get_probabilities()
        # For probabilities, mask out near-zero values
        threshold = 0.001  # Probabilities below 0.1% are masked

    vocab = tracking_processor.get_vocab_mapping()

    # Determine positions to plot
    if positions is None:
        positions = list(range(things['unstructured'].shape[1]))

    # Get indices of top k tokens (by maximum probability across all positions)
    max_probs = np.maximum(
        things['unstructured'].max(axis=1),
        things['structured'].max(axis=1)
    )
    top_indices = np.argsort(max_probs)[-k:]

    # Create masked arrays for better visualization
    def mask_array(arr):
        if kind == "logits":
            return np.ma.masked_where(arr < threshold, arr)
        else:
            return np.ma.masked_where(arr < threshold, arr)

    unstructured_masked = mask_array(things['unstructured'][top_indices][:, positions])
    structured_masked = mask_array(things['structured'][top_indices][:, positions])

    # Create figure
    if show_both:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        fig.suptitle(f'Token {kind.capitalize()} Evolution', fontsize=16, y=1.05)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 10))

    # Plot natural probabilities with masked array
    im1 = ax1.imshow(
        unstructured_masked,
        aspect='auto',
        cmap='viridis',
    )
    ax1.set_title(f'Natural Token {kind.capitalize()}')
    ax1.set_xlabel('Position in Sequence')
    ax1.set_ylabel('Token')
    if show_tokens:
        ax1.set_yticks(range(len(top_indices)))
        ax1.set_yticklabels([vocab[i][0] for i in top_indices])
    plt.colorbar(im1, ax=ax1, label=f'{kind.capitalize()}')

    # Plot constrained probabilities if requested
    if show_both:
        im2 = ax2.imshow(
            structured_masked,
            aspect='auto',
            cmap='viridis',
        )
        ax2.set_title(f'Constrained Token {kind.capitalize()}')
        ax2.set_xlabel('Position in Sequence')
        ax2.set_yticks([])  # Hide y-ticks since they're the same as ax1
        plt.colorbar(im2, ax=ax2, label=f'{kind.capitalize()}')

    plt.tight_layout()
    plt.savefig(f"{prefix}{kind}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


# This function applies a simple chat template to the prompt
def template(tokenizer, prompt: str, system_prompt: str = "You are a helpful assistant, responding in JSON.") -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )

def display_token_analysis(results, show_logits=True):
    """Helper to display token analysis results in a readable format."""
    for position_data in results:
        position_data['position']
        text = position_data['text_so_far']
        print(text)
        print("-" * 80)

        # Print header
        header = f"{'Token':<20} {'Natural Prob':<15} {'Constrained Prob':<15}"
        if show_logits:
            header += f"{'Natural Logit':<15} {'Constrained Logit':<15}"
        print(header)
        print("-" * 80)

        # Print each token's info
        for token_info in position_data['tokens']:
            # Add arrow for chosen token
            prefix = "→" if token_info['is_chosen'] else " "

            # Format probabilities as percentages (format first, then pad)
            unstructured_prob = f"{token_info['unstructured_prob']:.1%}"
            structured_prob = f"{token_info['structured_prob']:.1%}"

            # Build the line piece by piece
            line = f"{prefix} {repr(token_info['token']):<20} {unstructured_prob:<15} {structured_prob:<15}"

            # Add logits if requested
            if show_logits:
                unstructured_logit = f"{token_info['unstructured_logit']:.2f}"
                structured_logit = f"{token_info['structured_logit']:.2f}"
                line += f"{unstructured_logit:<15} {structured_logit:<15}"

            print(line)

def analyze_json_generation(model):
    """Analyze generation with JSON structure constraints."""
    print("\n=== Analyzing JSON-Structured Generation ===")

    # Define the required JSON structure
    class Person(BaseModel):
        name: str
        age: int
        zip_code: str = Field(pattern=r"^\d{5}$")
        state: str = Field(pattern=r"^[A-Z]{2}$")

    # Create generator with tracking
    generator = generate.json(model, Person)
    generator = track_logits(generator)

    # Generate JSON
    prompt = template(model.tokenizer.tokenizer, "Make me a person with a name, age, zip code, and state. Return the JSON only.")
    print(f"\nPrompt: {prompt}")
    result = generator(prompt)
    print(f"Generated JSON: {result}")

    # Show how constraints affect token choices
    print("\nAnalyzing token choices with JSON constraints:")
    print("1. Token generation analysis (showing probabilities and logits):")
    results = generator.logits_processor.get_top_tokens(k=5, positions=[0, 1, 2, 3, 4])
    display_token_analysis(results, show_logits=True)

    # Convert to dataframe
    df = generator.logits_processor.to_dataframe(show="probs", min_value=0.01)

    # Retrieve only the tokens that were chosen
    chosen = df[df.chosen]
    print(chosen)

    # Show sequence at different points
    print("\n2. Generation sequence at different points:")
    for pos in [5, 10, 15, 20]:
        print(f"\nFirst {pos} tokens: {repr(generator.logits_processor.sequence(pos))}")

    # Visualize how JSON structure affects probabilities
    print("\nCreating visualizations:")
    print("1. Bar plot comparing natural vs constrained probabilities")
    plot_token_distributions(generator.logits_processor, k=30, positions=[0, 1, 2], prefix="structured_")

    print("2. Heatmap showing probability evolution with/without constraints")
    plot_heatmap(
        generator.logits_processor,
        k=10000,
        kind="logits",
        prefix="structured_",
        show_both=True,
        show_tokens=False
    )

def main():
    print("Loading model and tokenizer...")

    model_uri = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = models.transformers(model_uri)

    # Run examples
    analyze_json_generation(model)

if __name__ == "__main__":
    main()

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
from outlines.processors import LogitTrackingProcessor
from outlines.processors.tracking import add_tracking
from utils import plot_token_distributions, template, plot_heatmap


def display_token_analysis(results, show_logits=True):
    """Helper to display token analysis results in a readable format."""
    for position_data in results:
        pos = position_data['position']
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

def analyze_json_generation(model, tokenizer):
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
    generator = add_tracking(generator)
    
    # Generate JSON
    prompt = template(tokenizer.tokenizer, "Make me a person with a name, age, zip code, and state. Return the JSON only.")
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
    chosen = df[df.selected]
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
    model = models.transformers("HuggingFaceTB/SmolLM2-135M-Instruct", device="cuda")
    tokenizer = model.tokenizer
    
    # Run examples
    analyze_json_generation(model, tokenizer)


if __name__ == "__main__":
    main() 
from transformers import AutoModelForCausalLM, AutoTokenizer

from outlines.types.dsl import Regex
from outlines.models.transformers import from_transformers
import outlines


model = from_transformers(
    AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini"),
    AutoTokenizer.from_pretrained("erwanf/gpt2-mini"),
)

generator = outlines.Generator(model, Regex(r"\d{3}"), backend="xgrammar")
# pickable both before and after a generation
generator.to_disk("logits_processor.pkl")
result = generator("Generate a 3-digit number:")
generator.to_disk("logits_processor.pkl")

new_generator = outlines.Generator(model, processor="logits_processor.pkl")
result = new_generator("Generate a 3-digit number:")
print(result)

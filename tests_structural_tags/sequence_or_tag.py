from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines.models.transformers import from_transformers
from outlines.types import StructuralTags

MODEL_NAME = "erwanf/gpt2-mini"

model = from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

output_type = StructuralTags("""{
    "type": "structural_tag",
    "format": {
        "type": "tag",
        "begin": "<start>",
        "content": {
            "type": "or",
            "elements": [
                {
                    "type": "sequence",
                    "elements": [
                        {
                            "type": "tag",
                            "begin": "<foo>",
                            "content": {"type": "any_text"},
                            "end": "</foo>"
                        },
                        {
                            "type": "const_string",
                            "value": "const_string_1"
                        }
                    ]
                },
                {
                    "type": "qwen_xml_parameter",
                    "json_schema": {
                        "type": "object",
                        "properties": {"mandatory": {"type": "boolean"}, "age": {"type": "integer"}},
                        "required": ["mandatory", "age"]
                    }
                }
            ]
        },
        "end": "</end>"
    }
}""")

prompt = f"""
You are a helpful assistant.
"""

result = model(prompt, output_type, max_new_tokens=100)
print(result)

#<start> <parameter=mandatory>
#
#true
#
#</parameter>
#
#<parameter=age>
#
#-1
#
#</parameter></end>

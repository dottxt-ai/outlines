import bentoml

MODEL_ID = "mistralai/Mistral-7B-v0.1"
BENTO_MODEL_TAG = MODEL_ID.lower().replace("/", "--")


def import_model(model_id, bento_model_tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    with bentoml.models.create(bento_model_tag) as bento_model_ref:
        tokenizer.save_pretrained(bento_model_ref.path)
        model.save_pretrained(bento_model_ref.path)


if __name__ == "__main__":
    import_model(MODEL_ID, BENTO_MODEL_TAG)

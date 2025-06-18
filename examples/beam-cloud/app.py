from beam import Image, endpoint, env

if env.is_remote():
    import outlines


# Pre-load models when the container first starts
def load_models():
    import outlines

    model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
    return model


@endpoint(
    name="outlines-serverless",
    gpu="A10G",
    cpu=1,
    memory="16Gi",
    on_start=load_models,
    image=Image().add_python_packages(
        ["outlines", "torch", "transformers", "accelerate"]
    ),
)
def predict(context, **inputs):
    default_prompt = """You are a sentiment-labelling assistant.
    Is the following review positive or negative?

    Review: This restaurant is just awesome!
    """

    prompt = inputs.get("prompt", default_prompt)

    # Unpack cached model from context
    model = context.on_start_value
    # Inference
    generator = outlines.generate.choice(model, ["Positive", "Negative"])
    answer = generator(prompt)
    return {"answer": answer}

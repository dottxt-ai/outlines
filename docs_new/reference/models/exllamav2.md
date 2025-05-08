# ExllamaV2

The `outlines.models.exllamav2` model requires a Logits Processor component for compatibility with Outlines structured generation. While ExLlamaV2 doesn't natively support this feature, a third-party fork provides the necessary functionality. You can install it with the following command:

```bash
pip install git+https://github.com/lapp0/exllamav2@sampler-logits-processor
```

Install other requirements:

```bash
pip install transformers torch
```

*Coming soon*

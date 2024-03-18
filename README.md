# quiet-star

This project is implemented by simply patching the base Mistral implementation in Huggingface `transformers` using a new `modeling_mistral.py` and a new `configuration_mistral.py` and otherwise applying standard features. Our patches were applied to Huggingface's `transformers` version `4.37.0.dev0` -- we cannot guarantee that other changes to their implementation will not affect our implementation, so for reproducibility, we encourage using the same version.

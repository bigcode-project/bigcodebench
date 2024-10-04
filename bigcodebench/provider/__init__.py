from bigcodebench.provider.base import DecoderBase


def make_model(
    model: str,
    backend: str,
    subset: str,
    split: str,
    dataset: str = "bigcodebench",
    temperature: float = 0.0,
    # instruction model only
    instruction_prefix=None,
    response_prefix=None,
    # vllm only
    tp=1,
    direct_completion=False,
    base_url=None,
    trust_remote_code=False,
    # hf only
    attn_implementation="eager",
    # tokenizer
    tokenizer_name=None,
    tokenizer_kwargs=None,
) -> DecoderBase:
    if backend == "vllm":
        from bigcodebench.provider.vllm import VllmDecoder

        return VllmDecoder(
            name=model,
            subset=subset,
            split=split,
            temperature=temperature,
            dataset=dataset,
            direct_completion=direct_completion,
            tensor_parallel_size=tp,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
    elif backend == "hf":
        from bigcodebench.provider.hf import HuggingFaceDecoder

        return HuggingFaceDecoder(
            name=model,
            subset=subset,
            split=split,
            temperature=temperature,
            dataset=dataset,
            direct_completion=direct_completion,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            attn_implementation=attn_implementation,
        )
    elif backend == "openai":
        from bigcodebench.provider.openai import OpenAIChatDecoder

        assert not direct_completion, f"{backend} backend does not serve base model"
        return OpenAIChatDecoder(
            name=model,
            subset=subset,
            split=split,
            temperature=temperature,
            base_url=base_url,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
    elif backend == "mistral":
        from bigcodebench.provider.mistral import MistralChatDecoder
        
        return MistralChatDecoder(
            name=model,
            subset=subset,
            split=split,
            temperature=temperature,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
    elif backend == "anthropic":
        from bigcodebench.provider.anthropic import AnthropicDecoder

        assert not direct_completion, f"{backend} backend does not serve base model"
        return AnthropicDecoder(
            name=model,
            subset=subset,
            split=split,
            temperature=temperature,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
    elif backend == "google":
        from bigcodebench.provider.google import GoogleDecoder

        assert not direct_completion, f"{backend} backend does not serve base model"
        return GoogleDecoder(
            name=model,
            subset=subset,
            split=split,
            temperature=temperature,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
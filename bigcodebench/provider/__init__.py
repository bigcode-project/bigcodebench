from bigcodebench.provider.base import DecoderBase


def make_model(
    model: str,
    backend: str,
    subset: str,
    split: str,
    dataset: str = "bigcodebench",
    temperature: float = 0.0,
    max_new_tokens: int = 1280,
    # instruction model only
    instruction_prefix: str = None,
    response_prefix: str = None,
    # vllm and hf only
    revision: str = "main",
    # vllm only
    tp: int = 1,
    direct_completion: bool = False,
    base_url: str = None,
    trust_remote_code: bool = False,
    # hf only
    attn_implementation: str = "eager",
    # tokenizer
    tokenizer_name: str = None,
    tokenizer_legacy: bool = True,
) -> DecoderBase:
    if backend == "vllm":
        from bigcodebench.provider.vllm import VllmDecoder

        return VllmDecoder(
            name=model,
            subset=subset,
            split=split,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            revision=revision,
            dataset=dataset,
            direct_completion=direct_completion,
            tp=tp,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            trust_remote_code=trust_remote_code,
            tokenizer_name=tokenizer_name,
            tokenizer_legacy=tokenizer_legacy,
        )
    elif backend == "hf":
        from bigcodebench.provider.hf import HuggingFaceDecoder

        return HuggingFaceDecoder(
            name=model,
            subset=subset,
            split=split,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            revision=revision,
            dataset=dataset,
            direct_completion=direct_completion,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            tokenizer_name=tokenizer_name,
            tokenizer_legacy=tokenizer_legacy,
        )
    elif backend == "openai":
        from bigcodebench.provider.openai import OpenAIChatDecoder

        assert not direct_completion, f"{backend} backend does not serve base model"
        return OpenAIChatDecoder(
            name=model,
            subset=subset,
            split=split,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
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
            max_new_tokens=max_new_tokens,
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
            max_new_tokens=max_new_tokens,
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
            max_new_tokens=max_new_tokens,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
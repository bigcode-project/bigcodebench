import os
from typing import List

import anthropic

from bigcodebench.gen.util import anthropic_request
from bigcodebench.provider.base import DecoderBase
from bigcodebench.provider.utility import make_raw_chat_prompt

class AnthropicDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        if not do_sample:
            assert batch_size == 1, "Sampling only supports batch size of 1"

        all_outputs = []
        for prompt in tqdm(prompts):
            outputs = []
            
            for _ in range(num_samples):
                message = anthropic_request.make_auto_request(
                    client=self.client,
                    model=self.name,
                    messages=[
                        {
                            "role": "user",
                            "content": make_raw_chat_prompt(
                                task_prompt=prompt,
                                subset=self.subset,
                                split=self.split,
                                instruction_prefix=self.instruction_prefix,
                                response_prefix=self.response_prefix,
                                tokenizer=None,
                            )
                        }
                    ],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    stop_sequences=self.eos,
                )
                outputs.append(message.content[0].text)
            all_outputs.append(outputs)
        return outputs

    def is_direct_completion(self) -> bool:
        return False